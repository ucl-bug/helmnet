import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import hardtanh
from random import randint, choice
import pytorch_lightning as pl
import numpy as np
from helmnet.dataloaders import get_dataset
from helmnet.spectral import LaplacianWithPML
from helmnet.utils import load_settings, log_wavefield
from helmnet.source import Source
from helmnet.replaybuffer import ReplayBuffer, Experience
from torch.optim.lr_scheduler import ReduceLROnPlateau


def getActivationFunction(
    act_function_name: str, features=None, end=False
) -> nn.Module:
    """Returns the activation function module given
    the name

    Args:
        act_function_name (str): Name of the activation function, case unsensitive

    Raises:
        NotImplementedError: Raised if the activation function is unknown

    Returns:
        nn.Module
    """
    if act_function_name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif act_function_name.lower() == "celu":
        return nn.CELU(inplace=True)
    elif act_function_name.lower() == "relu_batchnorm":
        if end:
            return nn.ReLU(inplace=True)
        else:
            return nn.Sequential(nn.ReLU(inplace=True), nn.BatchNorm2d(features))
        return nn.CELU(inplace=True)
    elif act_function_name.lower() == "tanh":
        return nn.Tanh()
    elif act_function_name.lower() == "prelu":
        return nn.PReLU()
    elif act_function_name.lower() == "gelu":
        return nn.GELU()
    elif act_function_name.lower() == "tanhshrink":
        return nn.Tanhshrink()
    elif act_function_name.lower() == "softplus":
        return nn.Softplus()
    elif act_function_name.lower() == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    else:
        err = "Unknown activation function {}".format(act_function_name)
        raise NotImplementedError(err)


class OutConv(nn.Module):
    """Outconvolution, consisting of a simple 2D convolution layer with kernel size 1"""

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => actFunction) * 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels=None,
        activation_fun="relu",
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            getActivationFunction(activation_fun, mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)


class CleanDoubleConv(nn.Module):
    """(convolution => actFunction) * 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels=None,
        activation_fun="relu",
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            getActivationFunction(activation_fun, mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResDoubleConv(nn.Module):
    """(convolution => actFunction) * 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels=None,
        activation_fun="relu",
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            getActivationFunction(activation_fun, mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x) + x


class ConvGRUCell(nn.Module):
    """
    Basic CGRU cell.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, bias):

        super(ConvGRUCell, self).__init__()

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.update_gate = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
        self.reset_gate = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.out_gate = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur = cur_state
        # data size is [batch, channel, height, width]
        x_in = torch.cat([input_tensor, h_cur], dim=1)
        update = torch.sigmoid(self.update_gate(x_in))
        reset = torch.sigmoid(self.reset_gate(x_in))
        x_out = torch.tanh(
            self.out_gate(torch.cat([input_tensor, h_cur * reset], dim=1))
        )
        h_new = h_cur * (1 - update) + x_out * update
        return h_new


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        state_size=2,
        activation_function="prelu",
        use_state=True,
        domain_size=0,
    ):
        super().__init__()
        self.state_size = state_size
        self.use_state = use_state
        self.domain_size = domain_size
        self.num_features = num_features

        # Define the two double_conv layers
        self.conv_signal = DoubleConv(
            self.num_features + self.state_size * self.use_state,
            self.num_features,
            activation_fun=activation_function,
        )

        #  Downward path
        self.down = nn.Conv2d(
            self.num_features, self.num_features, kernel_size=8, padding=3, stride=2
        )
        if self.use_state:
            self.conv_state = DoubleConv(
                self.num_features + self.state_size,
                self.state_size,
                activation_fun=activation_function,
            )
            """
            self.conv_state = ConvGRUCell(
                in_channels=self.num_features,
                hidden_channels=self.state_size,
                kernel_size=[3, 3],
                bias=True
            )
            """

        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def clear_state(self, x):
        self.state = torch.zeros(
            [x.shape[0], 2, self.domain_size, self.domain_size], device=x.device
        )

    def forward(self, x):
        if self.use_state:
            if self.state is None:
                raise ValueError(
                    "You must set or clear the state before using this module"
                )
            x_and_state = torch.cat([x, self.state], 1)
            output = self.conv_signal(x_and_state)
            self.state = self.conv_state(torch.cat([output, self.state], 1))
            # self.state = self.conv_state(output, self.state)
        else:
            output = self.conv_signal(x)
        return output, self.down(output)


class ResNet(nn.Module):
    def __init__(
        self,
        activation_function: str,
        depth: int,
        domain_size: int,
        features: int,
        inchannels: int,
        state_channels: int,
        state_depth: int,
    ):
        super().__init__()

        # Hyperparameters
        self.activation_function = activation_function
        self.depth = depth
        self.domain_size = domain_size
        self.features = features
        self.inchannels = inchannels
        self.state_channels = state_channels
        self.state_depth = state_depth
        self.state = None

        #  Define resnet
        inc = [nn.Conv2d(inchannels + 2, features, 7, padding=3)]
        res_blocks = [
            ResDoubleConv(features, features, features * 2) for _ in range(self.depth)
        ]
        outc = [nn.Conv2d(features, 4, 7, padding=3)]
        layers = inc + res_blocks + outc
        self.network = nn.Sequential(*layers)

    def init_by_size(self):
        return

    def get_states(self, flatten=False):
        return

    def clear_states(self, x):
        self.state = None
        return

    def set_states(self, states, flatten=False):
        return

    def flatten_state(self, h_list):
        return

    def unflatten_state(self, h_flatten):
        return

    def forward(self, x):
        if self.state is None:
            self.state = torch.zeros(
                (x.shape[0], 2, x.shape[2], x.shape[3]), device=x.device
            )
        x = torch.cat([x, self.state], 1)
        y = self.network(x)
        self.state = y[:, :2]
        return y[:, 2:]


class HybridNet(nn.Module):
    def __init__(
        self,
        activation_function: str,
        depth: int,
        domain_size: int,
        features: int,
        inchannels: int,
        state_channels: int,
        state_depth: int,
    ):
        super().__init__()
        # Hyperparameters
        self.activation_function = activation_function
        self.depth = depth
        self.domain_size = domain_size
        self.features = features
        self.inchannels = inchannels
        self.state_channels = state_channels
        self.state_depth = state_depth

        #  Define states boundaries for packing and unpacking
        self.init_by_size()

        # Input layer
        self.inc = DoubleConv(
            self.inchannels, self.features, activation_fun=self.activation_function
        )

        # Encoding layer
        self.enc = nn.ModuleList(
            [
                EncoderBlock(
                    self.features,
                    state_size=self.state_channels,
                    activation_function=self.activation_function,
                    use_state=d < self.state_depth,
                    domain_size=self.states_dimension[d],
                )
                for d in range(self.depth)
            ]
        )

        # Decode path
        self.decode = nn.ModuleList(
            [
                DoubleConv(
                    self.features + self.features * (i < self.depth),
                    self.features,
                    activation_fun=self.activation_function,
                )
                for i in range(self.depth + 1)
            ]
        )

        # Upsampling
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    self.features,
                    self.features,
                    kernel_size=8,
                    padding=3,
                    output_padding=0,
                    stride=2,
                )
                for i in range(self.depth)
            ]
        )

        # Output layer
        self.outc = OutConv(self.features, 2)

    def init_by_size(self):
        # This helps to reshape the state to the correct dimensions
        self.states_dimension = [self.domain_size // 2 ** x for x in range(self.depth)]
        self.total_state_length = sum(map(lambda x: x ** 2, self.states_dimension))
        self.state_boundaries = []
        for d in range(self.depth):
            if d == 0:
                self.state_boundaries.append([0, self.states_dimension[d] ** 2])
            else:
                self.state_boundaries.append(
                    [
                        self.state_boundaries[-1][-1],
                        self.state_boundaries[-1][-1] + self.states_dimension[d] ** 2,
                    ]
                )

    def get_states(self, flatten=False):
        h = []
        for enc in self.enc:
            h.append(enc.get_state())
        if flatten:
            return self.flatten_state(h)
        else:
            return h

    def clear_states(self, x):
        for enc in self.enc:
            enc.clear_state(x)

    def set_states(self, states, flatten=False):
        if flatten:
            h = self.unflatten_state(states)
        for enc, state in zip(self.enc[: len(h)], h):
            enc.set_state(state)

    def flatten_state(self, h_list):
        h = []
        for x in h_list:
            h.append(x.view(x.shape[0], x.shape[1], -1))
        return torch.cat(h, 2)

    def unflatten_state(self, h_flatten):
        h = []
        h_shape = h_flatten.shape
        for boundaries, size in zip(self.state_boundaries, self.states_dimension):
            h_d_flat = h_flatten[:, :, boundaries[0] : boundaries[1]]
            h.append(h_d_flat.view(h_shape[0], h_shape[1], size, size))
        return h

    def forward(self, x):

        # First feature transformation
        x = self.inc(x)

        # Downsampling tree and extracting new states
        inner_signals = []
        for d in range(self.depth):
            # Encode signal
            inner, x = self.enc[d](x)
            # Store signal
            inner_signals.append(inner)

        # Upscaling
        x = self.decode[-1](x)
        for d in range(self.depth - 1, -1, -1):
            # Upscale
            x = self.up[d](x)
            # Concatenate inner path
            x = torch.cat([x, inner_signals[d]], 1)
            # Decode
            x = self.decode[d](x)

        # Output layer
        out = self.outc(x)

        return out
