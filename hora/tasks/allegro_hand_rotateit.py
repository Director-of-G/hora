# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import torch
import numpy as np
import pytorch3d.transforms as p3dtf
from tqdm import tqdm
from pathlib import Path

from gym.utils import seeding
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, unscale, quat_apply, tensor_clamp, torch_rand_float, quat_conjugate, quat_mul
from glob import glob
from hora.utils.misc import tprint
from hora.utils.common import get_all_files_with_suffix, get_all_files_with_name, load_from_pickle, \
    get_filename_from_path
from hora.utils.isaac_utils import load_an_object_asset, load_a_goal_object_asset, load_obj_texture, \
    get_link_to_collision_geometry_map
from hora.utils.torch_utils import torch_float, torch_long, quat_xyzw_to_wxyz
from .base.vec_task import VecTask


class AllegroHandRotateIt(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.set_random_gen()
        self.object_urdfs, self.dataset_path, self.urdf_path, self.obj_name_to_cat_id = \
            self.parse_obj_dataset(config["env"]["object"]["dataset"])
        self.num_objects = len(self.object_urdfs)

        self.config = config
        # before calling init in VecTask, need to do
        # 1. setup randomization
        self._setup_domain_rand_config(config['env']['randomization'])
        # 2. setup privileged information
        self._setup_priv_option_config(config['env']['privInfo'])
        # 3. setup object assets
        self._setup_object_info(config['env']['object'])
        # 4. setup reward
        self._setup_reward_config(config['env']['reward'])
        self.base_obj_scale = config['env']['baseObjScale']
        self.save_init_pose = config['env']['genGrasps']
        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.reset_z_threshold = self.config['env']['reset_height_threshold']
        self.grasp_cache_name = self.config['env']['grasp_cache_name']
        self.grasp_cache_size = self.config['env']['grasp_cache_size']
        self.evaluate = self.config['on_evaluation']

        # obj_orientation, obj_angvel & obj_restitution are considered in RotateIt
        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_orientation': (3, 7),
            'obj_angvel': (7, 10),
            'obj_scale': (10, 11),
            'obj_mass': (11, 12),
            'obj_friction': (12, 13),
            'obj_com': (13, 16),
            'obj_restitution': (16, 17)
        }

        super().__init__(config, sim_device, graphics_device_id, headless)

        self.read_generated_ptd()

        self.debug_viz = self.config['env']['enableDebugVis']
        self.max_episode_length = self.config['env']['episodeLength']
        self.dt = self.sim_params.dt

        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # object apply random forces parameters
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        # useful buffers
        self.object_rot_prev = self.object_rot.clone()
        self.object_pos_prev = self.object_pos.clone()
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.torques = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain

        # debug and understanding statistics
        self.env_timeout_counter = to_torch(np.zeros((len(self.envs)))).long().to(self.device)  # max 10 (10000 envs)
        self.stat_sum_rewards = 0
        self.stat_sum_rotate_rewards = 0
        self.stat_sum_episode_length = 0
        self.stat_sum_obj_linvel = 0
        self.stat_sum_torques = 0
        self.env_evaluated = 0
        self.max_evaluate_envs = 500000

        self.num_parallel_steps = 0

    def read_generated_ptd(self):
        self.hand_ptd_dict = load_from_pickle(self.hand_ptd_path)  # body_link name to point cloud
        body_links = list(self.link_to_geom_map.keys())
        for link_name in body_links:
            if "base_link" in link_name:
                body_links.remove(link_name)
        tprint(f'Pre-generated point cloud file contains point cloud for the following links:')
        for link in body_links:
            print(f'       {link}')
        self.ptd_body_links = body_links
        self.hand_body_links_to_handles = self.gym.get_actor_rigid_body_dict(self.envs[0], self.dex_hands[0])

        self.hand_ptds = torch.from_numpy(
            np.stack([self.hand_ptd_dict[self.link_to_geom_map[x]] for x in self.ptd_body_links])
        )
        self.hand_ptds = self.hand_ptds.to(self.device)
        self.base_link_handle = torch_long([self.hand_body_links_to_handles['base_link']])
        self.hand_body_handles = [self.hand_body_links_to_handles[x] for x in self.ptd_body_links]
        self.hand_body_handles = torch_long(self.hand_body_handles, device=self.device)

        self.base_link_pose_inv_rot = None
        self.base_link_pose_inv_pos = None
        self.quantization_size = None
        self.finger_tip_links = ['link_3.0', 'link_7.0', 'link_11.0', 'link_15.0']

        self.hand_body_links_to_handles = self.gym.get_actor_rigid_body_dict(self.envs[0], self.dex_hands[0])
        self.base_link_handle = torch_long([self.hand_body_links_to_handles['base_link']])
        self.finger_tip_handles = [self.hand_body_links_to_handles[x] for x in self.finger_tip_links]
        self.finger_tip_handles = torch_long(self.finger_tip_handles, device=self.device)

        hand_ptds = self.hand_ptds.repeat(self.num_envs, 1, 1, 1)
        self.hand_cad_ptd = hand_ptds.view(-1, hand_ptds.shape[-2], hand_ptds.shape[-1]).float()
        self.obj_cad_ptd = self.object_ptds
        self.obj_cad_ptd = self.obj_cad_ptd.view(-1, self.obj_cad_ptd.shape[-2], self.obj_cad_ptd.shape[-1]).float()

        self.se3_T_buf = torch.eye(4, device=self.device).repeat(self.num_envs * (len(self.ptd_body_links) + 2),
                                                                 1,
                                                                 1)

        self.se3_T_hand_buf = torch.eye(4, device=self.device).repeat(self.num_envs * len(self.ptd_body_links),
                                                                      1,
                                                                      1)

        self.se3_T_obj_buf = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)

    def read_generated_grasp_poses(self, object_names):
        dataset_name = self.config['env']['object']['dataset']
        self.saved_grasping_states = {}
        grasp_cache_size = self.config['env']['grasp_cache_size']
        # this lut version is faster for reset_idx
        self.saved_grasping_states_lut = torch.zeros((0, grasp_cache_size, 23), dtype=torch.float, device=self.device)
        if self.randomize_scale and self.scale_list_init:
            for obj_name in object_names:
                if obj_name not in self.saved_grasping_states:
                    self.saved_grasping_states[obj_name] = {}
                for s in self.randomize_scale_list:
                    scale_key = str(round(s, 2)).replace(".", "")
                    try:
                        self.saved_grasping_states[obj_name][scale_key] = torch.from_numpy(np.load(
                            f'assets/{dataset_name}/cache/{obj_name}/grasp_50k_s{scale_key}.npy'
                        )).float().to(self.device)
                        self.saved_grasping_states_lut = torch.cat(
                            [self.saved_grasping_states_lut, self.saved_grasping_states[obj_name][scale_key].clone().unsqueeze(0)],
                            dim=0
                        )
                    except FileNotFoundError:
                        self.saved_grasping_states.pop(obj_name)
                        tprint(f'Grasping states for {obj_name} not found for scale {s}')
                        break
        else:
            assert self.save_init_pose

        return list(self.saved_grasping_states.keys())

    def set_random_gen(self, seed=12345):
        self.np_random, seed = seeding.np_random(seed)

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_hand_asset()
        object_names, object_assets, object_textures, object_ptds = self.load_object_asset()
        valid_object_names = self.read_generated_grasp_poses(object_names)
        keep_idx = []
        for idx, obj_name in enumerate(object_names):
            if obj_name in valid_object_names:
                keep_idx.append(idx)
        object_names = [object_names[i] for i in keep_idx]
        object_assets = [object_assets[i] for i in keep_idx]
        object_ids = list(range(len(keep_idx)))
        object_textures = [object_textures[i] for i in keep_idx]
        object_ptds = [object_ptds[i] for i in keep_idx]
        self.object_names = object_names

        # set allegro_hand dof properties
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            allegro_hand_dof_props['effort'][i] = 0.5
            if self.torque_control:
                allegro_hand_dof_props['stiffness'][i] = 0.
                allegro_hand_dof_props['damping'][i] = 0.
                allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:
                allegro_hand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                allegro_hand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
            allegro_hand_dof_props['friction'][i] = 0.01
            allegro_hand_dof_props['armature'][i] = 0.001

        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)

        hand_pose, obj_pose = self._init_object_pose()

        # compute aggregate size
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        # max_agg_bodies = self.num_allegro_hand_bodies + 2
        # max_agg_shapes = self.num_allegro_hand_shapes + 2

        self.dex_hands = []
        self.envs = []

        self.object_init_state = []

        self.hand_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        allegro_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_assets[0])
        self.object_rb_handles = list(range(allegro_hand_rb_count, allegro_hand_rb_count + object_rb_count))

        self.object_ptds = []
        self.object_handles = []
        num_object_assets = len(object_assets)
        env_obj_ids = []
        env_obj_scales = []
        env_obj_grasp_cache_idx = []
        for i in range(num_envs):
            obj_asset_id = i % num_object_assets
            env_obj_ids.append(object_ids[obj_asset_id])

            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                obj_num_bodies = self.gym.get_asset_rigid_body_count(object_assets[obj_asset_id])
                obj_num_shapes = self.gym.get_asset_rigid_shape_count(object_assets[obj_asset_id])
                max_agg_bodies = self.num_allegro_hand_bodies + obj_num_bodies * 2 + 1
                max_agg_shapes = self.num_allegro_hand_shapes + obj_num_shapes * 2 + 1
                # self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.dex_hands.append(hand_actor)

            # add object
            # object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            # object_asset = self.object_asset_list[object_type_id]
            object_asset = object_assets[obj_asset_id]

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 0)
            self.object_handles.append(object_handle)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            obj_scale = obj_scale_randomize = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale_id = (i // num_object_assets) % num_scales
                obj_scale = self.randomize_scale_list[obj_scale_id]
                obj_scale_randomize = np.random.uniform(obj_scale - 0.025, obj_scale + 0.025)
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale_randomize)
            self._update_priv_buf(env_id=i, name='obj_scale', value=obj_scale_randomize)
            env_obj_scales.append(obj_scale)

            # save grasp cache index
            grasp_cache_idx = obj_asset_id * num_scales + obj_scale_id
            env_obj_grasp_cache_idx.append(grasp_cache_idx)

            # scale pointcloud
            self.object_ptds.append(obj_scale_randomize * object_ptds[obj_asset_id])

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            obj_friction = 1.0
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)

            # restitution is included as priv_info in RotateIt, but not randomized
            obj_restitution = 0.0
            if self.randomize_restitution:
                rand_restitution = np.random.uniform(self.randomize_restitution_lower, self.randomize_restitution_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.restitution = rand_restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.restitution = rand_restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_restitution = rand_restitution
            self._update_priv_buf(env_id=i, name='obj_restitution', value=obj_restitution)

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)

            # set object color (or render texture if needed)
            color = np.array([179, 193, 134]) / 255.0
            self.gym.set_rigid_body_color(
                env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.env_obj_grasp_cache_idx = to_torch(env_obj_grasp_cache_idx, dtype=torch.long, device=self.device)

        self.env_obj_ids = torch_long(env_obj_ids, device=self.device).view(-1, 1)
        self.env_obj_scales = torch_float(env_obj_scales, device=self.device).view(-1, 1)
        self.object_ptds = np.stack(self.object_ptds, axis=0)
        self.object_ptds = torch_float(self.object_ptds, device=self.device)

    def parse_obj_dataset(self, dataset):
        asset_root = Path(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', 'assets')
        ).resolve()
        split_dataset_name = dataset.split(':')
        dataset_path = asset_root.joinpath(dataset)
        if len(split_dataset_name) == 1:
            urdf_path = asset_root.joinpath(dataset, 'urdf')
        else:
            target_object = split_dataset_name[1]
            urdf_path = asset_root.joinpath(split_dataset_name[0], 'urdf', target_object)

        tprint(f'URDF path:{urdf_path}\nDataset path:{dataset_path}')
        urdf_files = get_all_files_with_suffix(urdf_path, suffix="urdf")
        permute_ids = self.np_random.permutation(np.arange(len(urdf_files)))
        permuted_urdfs = [urdf_files[i] for i in permute_ids]
        object_names = [os.path.splitext(posixpath.name)[0] for posixpath in permuted_urdfs]
        obj_name_to_id = {name: idx for idx, name in enumerate(object_names)}
        return permuted_urdfs, dataset_path, urdf_path, obj_name_to_id

    def load_object_asset(self):
        asset_root = Path(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', 'assets')).resolve()
        object_urdfs = self.object_urdfs

        object_names, object_assets, object_tex_handles, object_ptds = [], [], [], []
        # object_ids, object_cat_ids = [], []
        if "object_id" in self.config["env"]["object"]:
            urdf_to_load = self.object_urdfs[self.config["env"]["object"]["object_id"]]
            tprint(f'Loading a single object: {urdf_to_load}')
            obj_name, obj_asset, texture_handle, ptd = self.load_an_object(asset_root,
                                                                           urdf_to_load)
            object_names.append(obj_name)
            object_assets.append(obj_asset)
            # object_ids.append(self.object_urdfs.index(urdf_to_load))
            object_tex_handles.append(texture_handle)
            object_ptds.append(ptd)
            # object_cat_ids.append(self.obj_name_to_cat_id[self.get_object_category(urdf_to_load)])
        else:
            if "start_id" not in self.config["env"]["object"]:
                start = 0
                end = min(len(object_urdfs), self.config["env"]["object"]["num_objs"])
            else:
                start = self.config["env"]["object"]["start_id"]
                end = min(start + self.config["env"]["object"]["num_objs"], len(object_urdfs))
            iters = range(start, end)
            tprint(f'Loading object IDs from {start} to {end}.')
            for idx in tqdm(iters, desc='Loading Asset'):
                urdf_to_load = object_urdfs[idx]
                obj_name, obj_asset, texture_handle, ptd = self.load_an_object(asset_root,
                                                                               urdf_to_load)
                object_names.append(obj_name)
                object_assets.append(obj_asset)
                # object_ids.append(self.object_urdfs.index(urdf_to_load))
                object_tex_handles.append(texture_handle)
                object_ptds.append(ptd)
                # object_cat_ids.append(self.obj_name_to_cat_id[self.get_object_category(urdf_to_load)])
        # return object_assets, goal_assets, object_ids, object_tex_handles, object_ptds, object_cat_ids
        return object_names, object_assets, object_tex_handles, object_ptds
    
    def load_an_object(self, asset_root, object_urdf):
        out = []
        obj_asset = load_an_object_asset(self.gym, self.sim, asset_root, object_urdf, vhacd=self.config['env']['vhacd'])
        # obj_asset = self.change_obj_asset_dyn(obj_asset)
        # goal_obj_asset = load_a_goal_object_asset(self.gym, self.sim, asset_root, object_urdf, vhacd=False)
        ptd = None

        if self.config["env"]["object"]["dataset"] == "ycb":
            mid_folder = "google_16k"
        elif self.config["env"]["object"]["dataset"] == "miscnet":
            mid_folder = ""

        object_name = get_filename_from_path(object_urdf, with_suffix=False)
        out.append(object_name)
        if self.config["env"]["loadCADPTD"]:
            ptd_file = object_urdf.parent.parent.joinpath(
                "meshes", object_name, mid_folder, f'point_cloud_{self.config["env"]["objCadNumPts"]}_pts.pkl')
            if ptd_file.exists():
                ptd = load_from_pickle(ptd_file)
        out.append(obj_asset)
        # out.append(goal_obj_asset)
        if self.config["env"]["object"]["load_texture"]:
            texture_handle = load_obj_texture(self.gym, self.sim, object_urdf)
            out.append(texture_handle)
        else:
            out.append([])
        out.append(ptd)
        return out

    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # slower version
        # ----------------------------------------
        # for env_id in env_ids:
        #     obj_id, obj_scale = self.env_obj_ids[env_id].item(), self.env_obj_scales[env_id].item()
        #     name_key = self.object_names[obj_id]
        #     scale_key = str(round(obj_scale, 2)).replace(".", "")
        #     sampled_pose_idx = np.random.randint(self.saved_grasping_states[name_key][scale_key].shape[0])
        #     # print(f"env {env_id} | reset {sampled_pose_idx}")
        #     sampled_pose = self.saved_grasping_states[name_key][scale_key][sampled_pose_idx].clone()
        #     self.root_state_tensor[self.object_indices[env_id], :7] = sampled_pose[16:]
        #     self.root_state_tensor[self.object_indices[env_id], 7:13] = 0
        #     pos = sampled_pose[:16]
        #     self.allegro_hand_dof_pos[env_id, :] = pos
        #     self.allegro_hand_dof_vel[env_id, :] = 0
        #     self.prev_targets[env_id, :self.num_allegro_hand_dofs] = pos
        #     self.cur_targets[env_id, :self.num_allegro_hand_dofs] = pos
        #     self.init_pose_buf[env_id, :] = pos.clone()
        # ----------------------------------------

        # faster version
        # ----------------------------------------
        grasp_cache_idx = self.env_obj_grasp_cache_idx[env_ids]
        sampled_pose_idx = np.random.randint(self.saved_grasping_states_lut[0].shape[0], size=len(env_ids))
        sampled_pose = self.saved_grasping_states_lut[grasp_cache_idx, sampled_pose_idx]
        self.root_state_tensor[self.object_indices[env_ids], :7] = sampled_pose[:, 16:]
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = 0
        pos = sampled_pose[:, :16]
        self.allegro_hand_dof_pos[env_ids, :] = pos
        self.allegro_hand_dof_vel[env_ids, :] = 0
        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.init_pose_buf[env_ids, :] = pos.clone()
        # ----------------------------------------

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def compute_observations(self):
        self._refresh_gym()
        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (torch.rand(self.allegro_hand_dof_pos.shape) * 2.0 - 1.0) * self.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix.to(self.device) + self.allegro_hand_dof_pos, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = unscale(
            self.allegro_hand_dof_pos[at_reset_env_ids], self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = self.allegro_hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        t_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()

        self.obs_buf[:, :t_buf.shape[1]] = t_buf
        self.at_reset_buf[at_reset_env_ids] = 0

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())
        # obj_orientation & obj_angvel are included as priv_info
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_orientation', value=self.object_rot.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_angvel', value=self.object_angvel.clone())

        # compute ptd observation
        self.scene_ptd_buf = self.compute_ptd_observations()

    def compute_ptd_observations(self):
        self.hand_link_pos = self.rigid_body_states[:, self.hand_body_handles][:, :, 0:3]
        self.hand_link_quat = self.rigid_body_states[:, self.hand_body_handles][:, :, 3:7]
        object_pos = self.object_pos
        object_quat = self.object_rot

        quats = torch.cat((self.hand_link_quat, object_quat[:, None, :]), dim=1)
        trans = torch.cat((self.hand_link_pos, object_pos[:, None, :]), dim=1)
        quats_in_p3d = quat_xyzw_to_wxyz(quats)
        rot_mat = p3dtf.quaternion_to_matrix(quats_in_p3d)
        if self.config['env']['ptd_to_robot_base']:
            if self.base_link_pose_inv_rot is None:
                base_link_pos = self.rigid_body_states[:, self.base_link_handle][..., :3]
                base_link_quat = self.rigid_body_states[:, self.base_link_handle][..., 3:7]
                base_link_quat_in_p3d = quat_xyzw_to_wxyz(base_link_quat)
                base_link_rot_mat = p3dtf.quaternion_to_matrix(base_link_quat_in_p3d)
                self.base_link_pose_inv_rot = base_link_rot_mat.transpose(-2, -1)
                self.base_link_pose_inv_pos = -self.base_link_pose_inv_rot @ base_link_pos.unsqueeze(-1)
            composed_rot = self.base_link_pose_inv_rot @ rot_mat
            composed_pos = self.base_link_pose_inv_rot @ trans.unsqueeze(-1) + self.base_link_pose_inv_pos
            rot_mat = composed_rot
            trans = composed_pos.squeeze(-1)

        rot_mat_T = rot_mat.transpose(-2, -1)
        self.se3_T_hand_buf[:, :3, :3] = rot_mat_T[:, :-1, :3, :3].reshape(-1, 3, 3)
        self.se3_T_hand_buf[:, 3, :3] = trans[:, :-1].reshape(-1, 3)
        self.se3_T_obj_buf[:, :3, :3] = rot_mat_T[:, -1, :3, :3].reshape(-1, 3, 3)
        self.se3_T_obj_buf[:, 3, :3] = trans[:, -1].reshape(-1, 3)
        hand_transform = p3dtf.Transform3d(matrix=self.se3_T_hand_buf)
        obj_transform = p3dtf.Transform3d(matrix=self.se3_T_obj_buf)

        hand_obs = hand_transform.transform_points(points=self.hand_cad_ptd)
        obj_obs = obj_transform.transform_points(points=self.obj_cad_ptd)
        hand_obs = hand_obs.view(self.num_envs, -1, 3)
        obj_obs = obj_obs.view(self.num_envs, -1, 3)
        if self.config['env']['include_robot_ptd']:
            ptd_obs = torch.cat((hand_obs, obj_obs), dim=1)
        else:
            ptd_obs = obj_obs

        if self.config['env']['debug']['visPcdObservation'] and self.num_parallel_steps % 10 == 0:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            ptd_obs_with_robot = torch.cat((hand_obs, obj_obs), dim=1)
            for i in range(ptd_obs_with_robot.shape[0]):
                pcd.points = o3d.utility.Vector3dVector(ptd_obs_with_robot[i].cpu().numpy().reshape(-1, 3))
                o3d.visualization.draw_geometries([pcd])

        if self.quantization_size is not None:
            ptd_obs = ptd_obs / self.quantization_size
            ptd_obs = ptd_obs.int()
        
        return ptd_obs

    def compute_reward(self, actions):
        self.rot_axis_buf[:, -1] = -1
        # pose diff penalty
        pose_diff_penalty = ((self.allegro_hand_dof_pos - self.init_pose_buf) ** 2).sum(-1)
        # work and torque penalty
        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2
        # Compute offset in radians. Radians -> radians / sec
        angdiff = quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.dt)
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)
        # linear velocity: use position difference instead of self.object_linvel
        object_linvel = ((self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)).clone()
        object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)
        # rotate penalty: alleviate unstable behaviors when rotating over x and y-axis
        rotate_penalty = torch.norm(torch.cross(object_angvel, self.rot_axis_buf, dim=-1), p=1, dim=-1)

        self.rew_buf[:] = compute_hand_reward(
            object_linvel_penalty, self.object_linvel_penalty_scale,
            rotate_reward, self.rotate_reward_scale,
            pose_diff_penalty, self.pose_diff_penalty_scale,
            torque_penalty, self.torque_penalty_scale,
            work_penalty, self.work_penalty_scale,
            rotate_penalty, self.rotate_penalty_scale
        )
        self.reset_buf[:] = self.check_termination(self.object_pos)
        self.extras['rotation_reward'] = rotate_reward.mean()
        self.extras['object_linvel_penalty'] = object_linvel_penalty.mean()
        self.extras['pose_diff_penalty'] = pose_diff_penalty.mean()
        self.extras['work_done'] = work_penalty.mean()
        self.extras['torques'] = torque_penalty.mean()
        self.extras['roll'] = object_angvel[:, 0].mean()
        self.extras['pitch'] = object_angvel[:, 1].mean()
        self.extras['yaw'] = object_angvel[:, 2].mean()
        self.extras['rotation_penalty'] = rotate_penalty.mean()

        if self.evaluate:
            finished_episode_mask = self.reset_buf == 1
            self.stat_sum_rewards += self.rew_buf.sum()
            self.stat_sum_rotate_rewards += rotate_reward.sum()
            self.stat_sum_torques += self.torques.abs().sum()
            self.stat_sum_obj_linvel += (self.object_linvel ** 2).sum(-1).sum()
            self.stat_sum_episode_length += (self.reset_buf == 0).sum()
            self.env_evaluated += (self.reset_buf == 1).sum()
            self.env_timeout_counter[finished_episode_mask] += 1
            info = f'progress {self.env_evaluated} / {self.max_evaluate_envs} | ' \
                   f'reward: {self.stat_sum_rewards / self.env_evaluated:.2f} | ' \
                   f'eps length: {self.stat_sum_episode_length / self.env_evaluated:.2f} | ' \
                   f'rotate reward: {self.stat_sum_rotate_rewards / self.env_evaluated:.2f} | ' \
                   f'lin vel (x100): {self.stat_sum_obj_linvel * 100 / self.stat_sum_episode_length:.4f} | ' \
                   f'command torque: {self.stat_sum_torques / self.stat_sum_episode_length:.2f}'
            tprint(info)
            if self.env_evaluated >= self.max_evaluate_envs:
                exit()

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        self.compute_reward(self.actions)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        # actions = torch.zeros_like(actions)
        self.actions = actions.clone().to(self.device)
        targets = self.prev_targets + 1 / 24 * self.actions
        self.cur_targets[:] = tensor_clamp(targets, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.prev_targets[:] = self.cur_targets.clone()
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
                 env in self.envs], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['mesh_ptd'] = self.scene_ptd_buf.to(self.rl_device)
        return self.obs_dict

    def step(self, actions):
        super().step(actions)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['mesh_ptd'] = self.scene_ptd_buf.to(self.rl_device)
        self.num_parallel_steps += 1
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def update_low_level_control(self):
        previous_dof_pos = self.allegro_hand_dof_pos.clone()
        self._refresh_gym()
        if self.torque_control:
            dof_pos = self.allegro_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
            self.torques = torch.clip(torques, -0.5, 0.5).clone()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def check_termination(self, object_pos):
        resets = torch.logical_or(
            torch.less(object_pos[:, -1], self.reset_z_threshold),
            torch.greater_equal(self.progress_buf, self.max_episode_length),
        )
        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        self.randomize_mass_lower = rand_config['randomizeMassLower']
        self.randomize_mass_upper = rand_config['randomizeMassUpper']
        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        self.randomize_friction = rand_config['randomizeFriction']
        self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_restitution = False
        self.randomize_restitution_lower = 0.0
        self.randomize_restitution_upper = 1.0
        self.randomize_scale = rand_config['randomizeScale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.joint_noise_scale = rand_config['jointNoiseScale']

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_orientation = p_config['enableObjOrientation']
        self.enable_priv_obj_angvel = p_config['enableObjAngVel']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']
        self.enable_priv_obj_restitution = p_config['enableObjRestitution']

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.priv_info_dict[name]
        if eval(f'self.enable_priv_{name}'):
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            if type(lower) is list or upper is list:
                lower = to_torch(lower, dtype=torch.float, device=self.device)
                upper = to_torch(upper, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0

    def _setup_object_info(self, o_config):
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)

        primitive_list = self.object_type.split('+')
        print('---- Primitive List ----')
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            # 'simple_tennis_ball': 'assets/ball.urdf',
            'simple_tennis_ball': 'assets/ycb/056_tennis_ball.urdf',
            'plastic_lemon': 'assets/ycb/014_lemon.urdf',
            'plastic_pear': 'assets/ycb/016_pear.urdf'
        }
        for p_id, prim in enumerate(primitive_list):
            if 'cuboid' in prim:
                subset_name = self.object_type.split('_')[-1]
                cuboids = sorted(glob(f'../assets/cuboid/{subset_name}/*.urdf'))
                cuboid_list = [f'cuboid_{i}' for i in range(len(cuboids))]
                self.object_type_list += cuboid_list
                for i, name in enumerate(cuboids):
                    self.asset_files_dict[f'cuboid_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cuboid_list) for _ in cuboid_list]
            elif 'cylinder' in prim:
                subset_name = self.object_type.split('_')[-1]
                cylinders = sorted(glob(f'assets/cylinder/{subset_name}/*.urdf'))
                cylinder_list = [f'cylinder_{i}' for i in range(len(cylinders))]
                self.object_type_list += cylinder_list
                for i, name in enumerate(cylinders):
                    self.asset_files_dict[f'cylinder_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cylinder_list) for _ in cylinder_list]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----')
        print(self.object_type_list)
        assert (len(self.object_type_list) == len(self.object_type_prob))

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config['env']['hora']['propHistoryLen']
        self.num_env_factors = self.config['env']['hora']['privInfoDim']
        self.priv_info_buf = torch.zeros((num_envs, self.num_env_factors), device=self.device, dtype=torch.float)
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, 32), device=self.device, dtype=torch.float)

    def _setup_reward_config(self, r_config):
        self.angvel_clip_min = r_config['angvelClipMin']
        self.angvel_clip_max = r_config['angvelClipMax']
        self.rotate_reward_scale = r_config['rotateRewardScale']
        self.object_linvel_penalty_scale = r_config['objLinvelPenaltyScale']
        self.pose_diff_penalty_scale = r_config['poseDiffPenaltyScale']
        self.torque_penalty_scale = r_config['torquePenaltyScale']
        self.work_penalty_scale = r_config['workPenaltyScale']
        self.rotate_penalty_scale = r_config['rotatePenaltyScale']

    def _create_hand_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.config['env']['asset']['handAsset']
        n_robot_cad_pts = self.config['env']['asset']['robotCadNumPts']
        self.hand_ptd_path = os.path.join(os.path.dirname(hand_asset_file), f'meshes/allegro/point_cloud_{n_robot_cad_pts}_pts.pkl')
        self.link_to_geom_map = get_link_to_collision_geometry_map(hand_asset_file)

        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = False    # set to False for synthetic pointcloud
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01

        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        else:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)

    # def _create_object_asset(self):
    #     # load object asset
    #     self.object_asset_list = []
    #     for object_type in self.object_type_list:
    #         object_asset_file = self.asset_files_dict[object_type]
    #         object_asset_options = gymapi.AssetOptions()
    #         object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
    #         self.object_asset_list.append(object_asset)

    def _init_object_pose(self):
        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(0, 0, 0.5)
        allegro_hand_start_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 1, 0), -np.pi / 2) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 2)
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_hand_start_pose.p.x
        pose_dx, pose_dy, pose_dz = -0.01, -0.04, 0.15

        object_start_pose.p.x = allegro_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = allegro_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_hand_start_pose.p.z + pose_dz

        object_start_pose.p.y = allegro_hand_start_pose.p.y - 0.01
        # for grasp pose generation, it is used to initialize the object
        # it should be slightly higher than the fingertip
        # so it is set to be 0.66 for internal allegro and 0.64 for the public allegro
        # ----
        # for in-hand object rotation, the initialization of z is only used in the first step
        # it is set to be 0.65 for backward compatibility
        object_z = 0.66 if self.save_init_pose else 0.65
        if 'internal' not in self.grasp_cache_name:
            object_z -= 0.02
        object_start_pose.p.z = object_z
        return allegro_hand_start_pose, object_start_pose


def compute_hand_reward(
    object_linvel_penalty, object_linvel_penalty_scale: float,
    rotate_reward, rotate_reward_scale: float,
    pose_diff_penalty, pose_diff_penalty_scale: float,
    torque_penalty, torque_pscale: float,
    work_penalty, work_pscale: float,
    rotate_penalty, rotate_penalty_scale: float,
):
    reward = rotate_reward_scale * rotate_reward
    # Distance from the hand to the object
    reward = reward + object_linvel_penalty * object_linvel_penalty_scale
    reward = reward + pose_diff_penalty * pose_diff_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + work_penalty * work_pscale
    reward = reward + rotate_penalty * rotate_penalty_scale
    return reward


def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 3:])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., :3] / sin_half_angles_over_angles
