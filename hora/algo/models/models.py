# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hora.algo.models.pointnet_utils import PointNetEncoderCustom as PointNetEncoder


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ProprioAdaptTConv(nn.Module):
    def __init__(self):
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(16 + 16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, 8)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x
    

class PointNetEncoderLight(nn.Module):
    def __init__(self, hidden_units=(32, 32, 32)):
        super(PointNetEncoderLight, self).__init__()
        self.hidden_units = hidden_units
        self.mlp_layers = self._build_mlp_layers()

    def _build_mlp_layers(self):
        layers = []
        input_dim = 3  # Assuming each point has (x, y, z) coordinates
        for i, hidden_dim in enumerate(self.hidden_units):
            layers.append(nn.Conv1d(input_dim, hidden_dim, 1))
            if i < len(self.hidden_units) - 1:
                layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # Transpose to (batch_size, 3, num_points)
        x = self.mlp_layers(x)  # Apply MLP
        x = torch.max(x, 2)[0]  # Max pooling over points
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.priv_mlp = kwargs.pop('priv_mlp_units')
        mlp_input_shape = input_shape[0]

        # point cloud encoder
        self.mesh_ptd = kwargs.pop('mesh_ptd')
        if self.mesh_ptd:
            mesh_mlp_units = kwargs.pop('mesh_mlp_units')
            mesh_emb_dim = mesh_mlp_units[-1]
            self.shape_encoder = PointNetEncoder(input_transform=False, mlp_units=mesh_mlp_units)
            # self.shape_encoder = PointNetEncoderLight(hidden_units=mesh_mlp_units)
        else:
            mesh_emb_dim = 0

        out_size = self.units[-1]
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['proprio_adapt']
        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1] + mesh_emb_dim
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'])

            if self.priv_info_stage2:
                self.adapt_tconv = ProprioAdaptTConv()

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1), # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        extrin, extrin_gt, mesh_emb = None, None, None
        if self.priv_info:
            if self.priv_info_stage2:
                extrin = self.adapt_tconv(obs_dict['proprio_hist'])
                # during supervised training, extrin has gt label
                extrin_gt = self.env_mlp(obs_dict['priv_info']) if 'priv_info' in obs_dict else extrin
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self.env_mlp(obs_dict['priv_info'])
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)

        if self.mesh_ptd:
            mesh_emb = self.shape_encoder(obs_dict['mesh_ptd'].transpose(-2, -1))[0]
            mesh_emb = torch.tanh(mesh_emb)
            obs = torch.cat([obs, mesh_emb], dim=-1)

        x = self.actor_mlp(obs)
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value, extrin, extrin_gt, mesh_emb

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt, mesh_emb = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
            'mesh_emb': mesh_emb
        }
        return result


if __name__ == "__main__":
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            torch.manual_seed(42)  # 设置随机种子
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    device = "cuda"

    batch_size = 16
    num_points = 1024
    point_cloud = torch.randn(batch_size, num_points, 3).to(device)  # Example point cloud data

    encoder_1 = PointNetEncoderLight()
    encoder_1.apply(init_weights)
    encoder_1.to(device)
    features_1 = encoder_1(point_cloud)
    print("Encoded features shape:", features_1.shape)  # Should be (batch_size, 32)
    print(encoder_1)

    from pointnet_utils import PointNetEncoderCustom as PointNetEncoder
    encoder_2 = PointNetEncoder(input_transform=False, mlp_units=[32, 32, 32])
    encoder_2.apply(init_weights)
    encoder_2.to(device)
    features_2 = encoder_2(point_cloud.transpose(-2, -1))[0]
    print("Encoded features shape:", features_2.shape)  # Should be (batch_size, 1024)
    print(encoder_2)

    print("all close between two nn outputs: {allclose}".format(allclose=torch.allclose(features_1, features_2)))  # Should be True
