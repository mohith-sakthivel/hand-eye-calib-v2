import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from deep_hand_eye.utils import unbatch


def make_conv_block(
    input_dim: int,
    feat_dim: int,
    output_dim: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Module:
    block = nn.Sequential(
        nn.Conv2d(input_dim, feat_dim, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(feat_dim, output_dim, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((3, 3)),
    )

    return block


class SelfAttention(nn.Module):
    def __init__(
        self,
        value_net: nn.Module,
        key_net: nn.Module,
        query_net: nn.Module,
    ):

        super().__init__()

        self.value_net = value_net
        self.key_net = key_net
        self.query_net = query_net

    def forward(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        query = self.query_net(inputs)
        key = self.key_net(inputs)
        value = self.value_net(inputs)
        feat_shape = value.shape[-3:]

        query = unbatch(query.flatten(start_dim=-3), index)
        key = unbatch(key.flatten(start_dim=-3), index)
        value = unbatch(value.flatten(start_dim=-3), index)

        attn = F.softmax(
            torch.matmul(query, key.permute(0, 2, 1)) / np.sqrt(2 * query.shape[-1]),
            dim=-1,
        )
        output = torch.matmul(attn, value).view(-1, *feat_shape)

        return output


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
