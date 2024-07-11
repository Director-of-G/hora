import os
import shutil
import subprocess

import open3d as o3d
import pybullet as p
import numpy as np
import pickle
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


WDH_RATIO_THRESH = 2.0      # filter criterion
DESIRED_OBJ_SCALE = 0.08      # desired object scale (max w/d/h, in mm) 


def generate_urdf_element_tree(urdf_name, visual_file, collision_file, scale):
    # 创建根元素
    robot = ET.Element("robot", name=urdf_name)

    # 创建link元素
    link = ET.SubElement(robot, "link", name="object")

    # 创建visual元素及其子元素
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", rpy="0 0 0", xyz="0 0 0")
    geometry_visual = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry_visual, "mesh", filename=visual_file, scale="{scale} {scale} {scale}".format(scale=scale))

    # 创建collision元素及其子元素
    collision = ET.SubElement(link, "collision")
    geometry_collision = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry_collision, "mesh", filename=collision_file, scale="{scale} {scale} {scale}".format(scale=scale))
    ET.SubElement(collision, "origin", rpy="0 0 0", xyz="0 0 0")

    # 创建inertial元素及其子元素
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="0.05")
    ET.SubElement(inertial, "inertia", ixx="0.0001", ixy="0.0", ixz="0.0", iyy="0.0001", iyz="0.0", izz="0.0001")

    # 将元素树转换为字符串
    tree = ET.ElementTree(robot)

    return tree


def process_dataset(dataset, exclude_objs, n_p):
    """
        Sample surface points, generate URDF for 3d meshes
        :param dataset: YCB
        :param folder: folder containing the meshes
        :param exclude_objs: objects to exclude
        :param n_p: number of points to sample
    """
    folder = f"./{dataset}/meshes"
    valid_objects = []

    if dataset == "ycb":
        ycb_object_names = os.listdir(folder)
        ycb_object_names.sort()
        for name in ycb_object_names:
            if name in exclude_objs:
                continue

            if os.path.exists(os.path.join(folder, "urdf")) and \
                name + '.urdf' in os.listdir(os.path.join(folder, "urdf")):
                print("URDF for {object} already exists!".format(object=name))
                continue

            # read mesh
            try:
                mesh = o3d.io.read_triangle_mesh(os.path.join(folder, name, "google_16k/nontextured.ply"))
                mesh.compute_vertex_normals()
                assert len(mesh.vertices) > 0
            except:
                print("Failed to read mesh for {object}".format(object=name))
                continue

            # compute width/depth/height ratio
            obb = mesh.get_oriented_bounding_box()
            obb_extent = obb.extent
            length, width, height = obb_extent[0], obb_extent[1], obb_extent[2]
            length_width_ratio = length / width
            length_height_ratio = length / height
            width_height_ratio = width / height
            min_ratio = min(length_width_ratio, length_height_ratio, width_height_ratio)
            max_ratio = max(length_width_ratio, length_height_ratio, width_height_ratio)
            if min_ratio <= 1/WDH_RATIO_THRESH or max_ratio >= WDH_RATIO_THRESH:
                print("Filter {object} due to WDH ratio".format(object=name))
                continue

            # sample surface point
            pcd = mesh.sample_points_poisson_disk(number_of_points=n_p)
            pcd_arr = np.asarray(pcd.points, dtype=np.float32)
            pickle.dump(
                pcd_arr,
                open(os.path.join(folder, name, "google_16k/point_cloud_{npts}_pts.pkl".format(npts=int(n_p))), "wb")
            )

            # generate v-hacd decomposition (with TestVHACD)
            textured_src = os.path.join(folder, name, "google_16k/textured.obj")
            textured_dst = os.path.join(folder, name, "google_16k/textured_vhacd.obj")
            # vhacd_result = subprocess.run(['TestVHACD', textured_src, '-g false'], stdout=subprocess.DEVNULL)
            # if vhacd_result.returncode != 0:
            #     print("V-HACD failed for {object}".format(object=name))
            #     continue
            # shutil.move("./decomp.obj", textured_dst)
            # os.remove("./decomp.mtl")
            # os.remove("./decomp.stl")

            # generate v-hacd decomposition (with trimesh)
            # try:
            #     mesh = trimesh.load(textured_src)
            #     convex_list = trimesh.interfaces.vhacd.convex_decomposition(mesh, debug=False)

            #     convex = trimesh.util.concatenate(convex_list)
            #     convex.export(textured_dst)
            # except ValueError:
            #     print("No direct VHACD backend available, trying pybullet")
            #     pass

            try:
                p.vhacd(textured_src, textured_dst, "/tmp/vhacd_log.txt")
            except ModuleNotFoundError:
                print('\n'+"ERROR - pybullet module not found: If you want to do convex decomposisiton, make sure you install pybullet (https://pypi.org/project/pybullet) or install VHACD directly (https://github.com/mikedh/trimesh/issues/404)"+'\n')
                raise

            # add mixing texture file
            mtl_file_src = "./ycb/056_tennis_ball/google_16k/textured.obj.mtl"
            mtl_file_dst = os.path.join(folder, name, "google_16k/textured.obj.mtl")
            if mtl_file_src != mtl_file_dst: shutil.copyfile(mtl_file_src, mtl_file_dst)
            texture_obj_file = os.path.join(folder, name, "google_16k/textured.obj")
            with open(texture_obj_file, 'r') as file:
                obj_content = file.readlines()
            for i in range(len(obj_content)):
                if obj_content[i].startswith('mtllib'):
                    obj_content[i] = f'mtllib textured.obj.mtl\n'
                    break
            with open(texture_obj_file, 'w') as file:
                file.writelines(obj_content)

            # generate URDF
            urdf_tree = generate_urdf_element_tree(
                urdf_name=name,
                visual_file=os.path.join(name, "google_16k/textured.obj"),
                collision_file=os.path.join(name, "google_16k/textured_vhacd.obj"),
                scale=DESIRED_OBJ_SCALE/max(length, width, height)
            )
            if not os.path.exists(os.path.join(folder, "urdf")):
                os.makedirs(os.path.join(folder, "urdf"))
            urdf_tree = ET.tostring(urdf_tree.getroot(), encoding='unicode')
            urdf_tree = minidom.parseString(urdf_tree)
            with open(os.path.join(folder, "urdf", "{name}.urdf".format(name=name)), "w") as file:
                file.write(urdf_tree.toprettyxml(indent="  "))

            # record object
            print("Successfully converted {object} to pointcloud and URDF!".format(object=name))
            valid_objects.append(name)

    elif dataset == "miscnet":
        miscnet_object_names = os.listdir(folder)
        miscnet_object_names.sort()
        for name in miscnet_object_names:
            if name in exclude_objs:
                continue

            if name == "urdf": continue

            if os.path.exists(os.path.join(folder, "urdf")) and \
                name + '.urdf' in os.listdir(os.path.join(folder, "urdf")):
                print("URDF for {object} already exists!".format(object=name))
                continue

            # copy urdf
            urdf_file_src = "./miscnet/meshes/{object}/model.urdf".format(object=name)
            urdf_file_dst = os.path.join(folder, "../urdf", "{object}.urdf".format(object=name))
            if not os.path.exists(os.path.join(folder, "urdf")):
                os.makedirs(os.path.join(folder, "urdf"))
            shutil.copyfile(urdf_file_src, urdf_file_dst)

            # change urdf content
            urdf_tree = ET.parse(urdf_file_dst)
            urdf_root = urdf_tree.getroot()
            for link in urdf_root.findall('.//link'):
                visual_geometry = link.find('visual/geometry/mesh')
                if visual_geometry is not None:
                    visual_geometry.set('filename', "meshes/{object}/visual_model.obj".format(object=name))

                collision_geometry = link.find('collision/geometry/mesh')
                if collision_geometry is not None:
                    collision_geometry.set('filename', "meshes/{object}/collision_model_vhacd.obj".format(object=name))

            urdf_tree.write(urdf_file_dst, encoding='utf-8', xml_declaration=True)

            # record object
            print("Successfully converted {object} to pointcloud and URDF!".format(object=name))
            valid_objects.append(name)


def generate_dexhand_ptd(hand_name="allegro", n_p=500):
    """
        Generate pointcloud for the dexterous hand
    """
    hand_mesh_folder = f"./{hand_name}/meshes/{hand_name}"
    hand_ptd_dict = {}
    for name in os.listdir(hand_mesh_folder):
        if not name.endswith("obj"):
            continue
        link_name = name.replace(".obj", "")

        mesh = o3d.io.read_triangle_mesh(os.path.join(hand_mesh_folder, name))
        mesh.compute_vertex_normals()

        pcd = mesh.sample_points_poisson_disk(number_of_points=n_p)
        pcd_arr = np.asarray(pcd.points, dtype=np.float32)

        hand_ptd_dict[link_name] = pcd_arr

    with open(os.path.join(hand_mesh_folder, f"./point_cloud_{n_p}_pts.pkl"), "wb") as f:
        pickle.dump(hand_ptd_dict, f)


if __name__ == "__main__":
    # process_dataset(dataset="miscnet", exclude_objs=[], n_p=100)
    generate_dexhand_ptd(hand_name="allegro", n_p=1000)
