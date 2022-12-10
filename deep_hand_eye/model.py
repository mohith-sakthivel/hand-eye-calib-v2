from typing import Tuple

import torch
import theseus as th
import torch.nn as nn
import torch.nn.functional as F

from deep_hand_eye.resnet import resnet18
from deep_hand_eye.theseus_opt import build_BA_Layer
from deep_hand_eye.pose_utils import qexp, xyz_quaternion_to_xyz_log_quaternion
from deep_hand_eye.layers import (
    make_conv_block,
    SelfAttention,
    SimpleConvEdgeUpdate,
)


class GCNet(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        gnn_recursion: int = 2,
        droprate: float = 0.0,
        pose_proj_dim: int = 32,
        rel_pose: bool = True,
    ) -> None:

        super().__init__()
        self.gnn_recursion = gnn_recursion
        self.droprate = droprate
        self.pose_proj_dim = pose_proj_dim
        self.rel_pose = rel_pose
        self.edge_feat_dim = edge_feat_dim

        # Setup the feature extractor
        self.feature_extractor = resnet18()
        self.process_feat = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, node_feat_dim, 3, 1, 0)
        )

        # Project relative robot displacement
        self.proj_rel_disp = nn.Linear(6, self.pose_proj_dim)
        # Intial edge project layer
        self.proj_init_edge = nn.Conv2d(
            in_channels=2 * node_feat_dim + pose_proj_dim,
            out_channels=edge_feat_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Setup the message passing network
        self.gnn_layer = SimpleConvEdgeUpdate(
            node_in_channels=node_feat_dim,
            node_out_channels=node_feat_dim,
            edge_in_channels=edge_feat_dim + pose_proj_dim,
            edge_out_channels=edge_feat_dim,
        )

        # Setup the relative pose regression networks
        if self.rel_pose:
            self.edge_R = nn.Sequential(
                nn.Conv2d(
                    in_channels=edge_feat_dim,
                    out_channels=edge_feat_dim,
                    kernel_size=3,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=edge_feat_dim,
                    out_channels=edge_feat_dim,
                    kernel_size=3,
                ),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((3, 3))
            )
            self.xyz_R = nn.Conv2d(
                in_channels=edge_feat_dim, out_channels=3, kernel_size=3
            )
            self.log_quat_R = nn.Conv2d(
                in_channels=edge_feat_dim, out_channels=3, kernel_size=3
            )

        # Setup the hand-eye pose regression networks
        # Self-attention for edges to transfer information
        def module_constructor() -> nn.Module:
            return make_conv_block(
                input_dim=edge_feat_dim + pose_proj_dim + 2 * node_feat_dim,
                feat_dim=edge_feat_dim,
                output_dim=edge_feat_dim,
                padding=0,
            )

        self.edge_self_attn_he = SelfAttention(
            value_net=module_constructor(),
            key_net=module_constructor(),
            query_net=module_constructor(),
        )

        # Attention to combine information from all edges
        self.edge_attn_he = nn.Conv2d(
            in_channels=edge_feat_dim,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        # Setup Regression heads
        self.xyz_he = nn.Conv2d(
            in_channels=edge_feat_dim, out_channels=3, kernel_size=3
        )
        self.log_quat_he = nn.Conv2d(
            in_channels=edge_feat_dim, out_channels=3, kernel_size=3
        )

        # Initialize networks
        init_modules = [
            self.proj_rel_disp,
            self.process_feat,
            self.proj_init_edge,
            self.gnn_layer,
            self.edge_self_attn_he,
            self.edge_attn_he,
            self.xyz_he,
            self.log_quat_he,
        ]
        if self.rel_pose:
            init_modules.extend([self.edge_R, self.xyz_R, self.log_quat_R])

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def join_node_edge_feat(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat_list: torch.Tensor,
    ) -> torch.Tensor:
        # Join node features of a corresponding edge
        out_feat = torch.cat(
            (
                node_feat[edge_index[0], ...],
                node_feat[edge_index[1], ...],
                *edge_feat_list,
            ),
            dim=-3,
        )
        return out_feat

    def forward(
        self, data: torch.Tensor, opt_iterations: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr = xyz_quaternion_to_xyz_log_quaternion(edge_attr)
        x = self.feature_extractor(x)
        x = self.process_feat(x)

        # Compute edge features
        rel_disp_feat = F.relu(self.proj_rel_disp(edge_attr), inplace=True)
        rel_disp_feat = rel_disp_feat.view(*rel_disp_feat.shape, 1, 1).expand(
            -1, -1, *x.shape[-2:]
        )
        edge_node_feat = self.join_node_edge_feat(x, edge_index, [rel_disp_feat])
        edge_feat = F.relu(self.proj_init_edge(edge_node_feat), inplace=True)

        # Graph message passing step
        for _ in range(self.gnn_recursion):
            edge_feat = torch.cat([edge_feat, rel_disp_feat], dim=-3)
            x, edge_feat = self.gnn_layer(x, edge_index, edge_feat)
            x = F.relu(x)
            edge_feat = F.relu(edge_feat)

        # Drop edge features if necessary
        if self.droprate > 0:
            edge_feat = F.dropout(edge_feat, p=self.droprate, training=self.training)

        # Predict the relative pose between images
        if self.rel_pose:
            edge_R_feat = self.edge_R(edge_feat)
            xyz_R = self.xyz_R(edge_R_feat).squeeze()
            log_quat_R = self.log_quat_R(edge_R_feat).squeeze()
            rel_pose_out = torch.cat((xyz_R, log_quat_R), dim=-1)
            C_T_C_pred = torch.cat((xyz_R, qexp(log_quat_R)), dim=-1).reshape(
                data.num_graphs, -1, 7
            )
        else:
            rel_pose_out = None

        # Process edge features for regressing hand-eye parameters
        edge_he_feat = self.join_node_edge_feat(
            x, edge_index, [edge_feat, rel_disp_feat]
        )
        # Find the graph id of each edge using the source node
        edge_graph_ids = data.batch[data.edge_index[0].cpu().numpy()]
        # Perform self-attention of edge features
        edge_he_feat = self.edge_self_attn_he(edge_he_feat, edge_graph_ids)

        # Calculate the attention weight over the edges
        edge_he_logits = (
            self.edge_attn_he(edge_he_feat).squeeze().repeat(data.num_graphs, 1)
        )
        num_graphs = (
            torch.arange(0, data.num_graphs).view(-1, 1).to(edge_graph_ids.device)
        )
        edge_he_logits[num_graphs != edge_graph_ids] = -torch.inf
        edge_he_attn = F.softmax(edge_he_logits, dim=-1)

        # Apply attention
        num_edges, feat_shape = edge_he_feat.shape[0], edge_he_feat.shape[1:]
        edge_he_aggr = torch.matmul(edge_he_attn, edge_he_feat.view(num_edges, -1))
        edge_he_aggr = edge_he_aggr.view(data.num_graphs, *feat_shape)

        # Predict the hand-eye parameters
        xyz_he = self.xyz_he(edge_he_aggr).reshape(-1, 3)
        log_quat_he = self.log_quat_he(edge_he_aggr).reshape(-1, 3)
        E_T_C_pred = torch.cat([xyz_he, qexp(log_quat_he)], dim=-1)

        # Perform differentiable non-linear optimization
        if opt_iterations > 0:
            processed_edge_index = edge_index.reshape(2, data.num_graphs, -1)
            edges_per_graph = num_edges // data.num_graphs
            processed_edge_index = (
                processed_edge_index.permute(1, 0, 2) % edges_per_graph
            )

            E_T_E_gt = torch.cat(
                [edge_attr[..., :3], qexp(edge_attr[..., 3:])], axis=-1
            )
            E_T_E_gt = E_T_E_gt.reshape(data.num_graphs, -1, 7)

            theseus_layer = build_BA_Layer(
                E_T_C_values=E_T_C_pred,
                C_T_C_values=C_T_C_pred,
                E_T_E_values=E_T_E_gt,
                edge_index=processed_edge_index,
                max_iterations=opt_iterations,
            )

            theseus_inputs = {"E_T_C": th.SE3(x_y_z_quaternion=E_T_C_pred)}

            for i in range(processed_edge_index.shape[-1]):
                edge_i, edge_j = (
                    processed_edge_index[0, 0, i],
                    processed_edge_index[0, 1, i],
                )
                # Create poses for camera-to-camera transforms
                theseus_inputs[f"C{edge_i}_T_C{edge_j}"] = th.SE3(
                    x_y_z_quaternion=C_T_C_pred[:, i]
                )
                # Create poses for transforms between robot end-effector positions
                theseus_inputs[f"E{edge_i}_T_E{edge_j}"] = th.SE3(
                    x_y_z_quaternion=E_T_E_gt[:, i]
                )

            updated_vars, info = theseus_layer.forward(
                input_tensors=theseus_inputs,
                optimizer_kwargs={"track_best_solution": False, "verbose": False},
            )

        else:
            updated_vars = {"E_T_C": th.SE3(x_y_z_quaternion=E_T_C_pred).to_matrix()}

        return updated_vars["E_T_C"], rel_pose_out, edge_index
