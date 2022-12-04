import numpy as np
from typing import Optional, Tuple

import torch
import theseus as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from deep_hand_eye.utils import unbatch
from deep_hand_eye.resnet import resnet34
from deep_hand_eye.pose_utils import qexp, xyz_quaternion_to_xyz_log_quaternion
from deep_hand_eye.theseus_opt import build_BA_Layer


class SimpleEdgeModel(nn.Module):
    """
    Network to perform autoregressive edge update during Neural message passing
    """

    def __init__(
        self, node_channels: int, edge_in_channels: int, edge_out_channels: int
    ):
        super().__init__()
        self.node_channels = node_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels

        self.edge_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * node_channels + edge_in_channels,
                out_channels=edge_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=edge_out_channels,
                out_channels=edge_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        out = torch.cat([edge_attr, source, target], dim=1).contiguous()
        out = self.edge_cnn(out)

        return out


class AttentionBlock(nn.Module):
    """
    Network to apply non-local attention
    """

    def __init__(self, in_channels: int, N: int = 8):
        super().__init__()
        self.N = N
        self.W_theta = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels // N, kernel_size=3
        )
        self.W_phi = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels // N, kernel_size=3
        )

        self.W_f = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // N,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.W_g = nn.Conv2d(
            in_channels=in_channels // N,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        out_channels = x.size(1)

        theta_x = self.W_theta(x)
        theta_x = F.adaptive_avg_pool2d(theta_x, 1).squeeze(dim=-1)
        phi_x = self.W_phi(x)
        phi_x = F.adaptive_avg_pool2d(phi_x, 1).squeeze(dim=-1)
        phi_x = phi_x.permute(0, 2, 1)
        W_ji = F.softmax(torch.matmul(theta_x, phi_x), dim=-1)

        t = self.W_f(x)
        t_shape = t.shape
        t = t.view(batch_size, out_channels // self.N, -1)
        t = torch.matmul(W_ji, t)
        t = t.view(*t_shape)
        a_ij = self.W_g(t)

        return x + a_ij


class SimpleConvEdgeUpdate(MessagePassing):
    """
    Network to pass messages and update the nodes
    """

    def __init__(
        self,
        node_in_channels: int,
        node_out_channels: int,
        edge_in_channels: int,
        edge_out_channels: int,
        use_attention: bool = True,
    ):
        super().__init__(aggr="mean")

        self.use_attention = use_attention

        self.edge_update_cnn = SimpleEdgeModel(
            node_channels=node_in_channels,
            edge_in_channels=edge_in_channels,
            edge_out_channels=edge_out_channels,
        )

        self.msg_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=node_in_channels + edge_out_channels,
                out_channels=node_out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=node_out_channels,
                out_channels=node_out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.node_update_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=node_in_channels + node_out_channels,
                out_channels=node_in_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=node_out_channels,
                out_channels=node_out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        if self.use_attention:
            self.att = AttentionBlock(in_channels=node_out_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        row, col = edge_index
        edge_attr = self.edge_update_cnn(x[row], x[col], edge_attr)

        # x has shape [N, in_channels] and edge_index has shape [2, E]
        H, W = x.shape[-2:]
        num_nodes, num_edges = x.shape[0], edge_attr.shape[0]
        out = self.propagate(
            edge_index=edge_index,
            size=(x.size(0), x.size(0)),
            x=x.view(num_nodes, -1),
            edge_attr=edge_attr.view(num_edges, -1),
            H=H,
            W=W,
        )
        return out, edge_attr

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        num_edges = edge_attr.shape[0]
        msg = self.msg_cnn(
            torch.cat(
                [x_j.view(num_edges, -1, H, W), edge_attr.view(num_edges, -1, H, W)],
                dim=-3,
            )
        )
        if self.use_attention:
            msg = self.att(msg)

        return msg.view(num_edges, -1)

    def update(
        self, aggr_out: torch.Tensor, x: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        num_nodes = x.shape[0]
        out = self.node_update_cnn(
            torch.cat(
                [x.view(num_nodes, -1, H, W), aggr_out.view(num_nodes, -1, H, W)],
                dim=-3,
            )
        )

        return out


class EdgeSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feat_dim: int,
        key_dim: Optional[int] = None,
        query_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
    ):

        super().__init__()

        self.feat_dim = feat_dim
        self.value_dim = value_dim if value_dim is not None else feat_dim
        self.key_dim = key_dim if key_dim is not None else feat_dim
        self.query_dim = query_dim if query_dim is not None else feat_dim

        self.value_net = self.make_conv_block(input_dim, self.feat_dim, self.value_dim)
        self.key_net = self.make_conv_block(input_dim, self.feat_dim, self.key_dim)
        self.query_net = self.make_conv_block(input_dim, self.feat_dim, self.query_dim)

    @staticmethod
    def make_conv_block(input_dim: int, feat_dim: int, output_dim: int) -> nn.Module:
        block = nn.Sequential(
            nn.Conv2d(input_dim, feat_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, output_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        return block

    def forward(
        self, edge_feat: torch.Tensor, edge_graph_id: torch.Tensor
    ) -> torch.Tensor:
        query = self.query_net(edge_feat)
        key = self.key_net(edge_feat)
        value = self.value_net(edge_feat)
        feat_shape = query.shape[-3:]

        query = unbatch(query.flatten(start_dim=-3), edge_graph_id)
        key = unbatch(key.flatten(start_dim=-3), edge_graph_id)
        value = unbatch(value.flatten(start_dim=-3), edge_graph_id)

        output = []
        attn = F.softmax(
            torch.matmul(query, key.permute(0, 2, 1)) / np.sqrt(2 * self.feat_dim),
            dim=-1,
        )
        output = torch.matmul(attn, value).view(-1, *feat_shape)

        return output


class GCNet(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 512,
        edge_feat_dim: int = 512,
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
        self.feature_extractor = resnet34(pretrained=True)
        self.process_feat = nn.Conv2d(
            in_channels=512,
            out_channels=node_feat_dim,
            kernel_size=3,
            stride=1,
            padding=1,
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
                    out_channels=edge_feat_dim // 2,
                    kernel_size=3,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=edge_feat_dim // 2,
                    out_channels=edge_feat_dim // 2,
                    kernel_size=3,
                ),
                nn.ReLU(inplace=True),
            )
            self.xyz_R = nn.Conv2d(
                in_channels=edge_feat_dim // 2, out_channels=3, kernel_size=3
            )
            self.log_quat_R = nn.Conv2d(
                in_channels=edge_feat_dim // 2, out_channels=3, kernel_size=3
            )

        # Setup the hand-eye pose regression networks
        # Self-attention for edges to transfer information
        self.edge_self_attn_he = EdgeSelfAttention(
            input_dim=edge_feat_dim + pose_proj_dim + 2 * node_feat_dim,
            feat_dim=edge_feat_dim // 2,
        )

        # Attention to combine information from all edges
        self.edge_attn_he = nn.Conv2d(
            in_channels=edge_feat_dim // 2,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        # Setup Regression heads
        self.xyz_he = nn.Conv2d(
            in_channels=edge_feat_dim // 2, out_channels=3, kernel_size=3
        )
        self.log_quat_he = nn.Conv2d(
            in_channels=edge_feat_dim // 2, out_channels=3, kernel_size=3
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
