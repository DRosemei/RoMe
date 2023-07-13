import torch
from torch import nn


class HeightMLP(nn.Module):
    def __init__(self, num_encoding, num_width):
        super().__init__()
        self.num_encoding = num_encoding
        self.D = num_width
        self.pos_channel = 2 * (2 * self.num_encoding + 1)
        self.height_layer_0 = nn.Sequential(
            nn.Linear(self.pos_channel, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
        )
        self.height_layer_1 = nn.Sequential(
            nn.Linear(self.D + self.pos_channel, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, 1),
        )

    def encode_position(self, input, levels, include_input=True):
        """
        For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
            - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
            itself results in 2L+1 elements.
            - With C channels, we get C(2L+1) channels output.

        :param input:   (..., C)            torch.float32
        :param levels:  scalar L            int
        :return:        (..., C*(2L+1))     torch.float32
        """

        # this is already doing "log_sampling" in the official code.
        result_list = [input] if include_input else []
        for i in range(levels):
            temp = 2.0**i * input  # (..., C)
            result_list.append(torch.sin(temp))  # (..., C)
            result_list.append(torch.cos(temp))  # (..., C)

        result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
        return result_list  # (..., C*(2L+1))

    def forward(self, norm_xy):
        encoded_norm_xy = self.encode_position(norm_xy, levels=self.num_encoding)
        feature_z = self.height_layer_0(encoded_norm_xy)
        vertices_z = self.height_layer_1(torch.cat([feature_z, encoded_norm_xy], dim=-1))
        return vertices_z
