import torch


class ResidualBlock(torch.nn.Module):
    """
    A bottle residual block, named as the number of channels is divided by 4 using a 1x1 conv before
    being expanded outwards again. This reduces the number of FLOPs used per block (compared to a vanilla residual block).
    """

    def __init__(self, c: int, k: int = 3):
        """
        Initializes Residual Block (bottleneck block)

        Args:
            c (int): Number of input channels
            k (int): Kernel size on middle convolutional layer. Defaults to 3 (as in paper).
        """
        super(ResidualBlock, self).__init__()
        assert c % 4 == 0
        self.conv1 = torch.nn.Conv2d(
            in_channels=c, out_channels=c // 4, kernel_size=1, stride=1
        )
        self.bn1 = torch.nn.BatchNorm2d(c // 4)
        self.conv2 = torch.nn.Conv2d(
            in_channels=c // 4, out_channels=c // 4, kernel_size=k, padding="same"
        )
        self.bn2 = torch.nn.BatchNorm2d(c // 4)
        self.conv3 = torch.nn.Conv2d(in_channels=c // 4, out_channels=c, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(c)
        self.relu = torch.nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        weights = (self.conv1.weight, self.conv2.weight, self.conv3.weight)
        biases = (self.conv1.bias, self.conv2.bias, self.conv3.bias)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="leaky_relu")
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def forward(self, x):
        cloned = x.clone()
        cloned = self.relu(self.bn1(self.conv1(cloned)))
        cloned = self.relu(self.bn2(self.conv2(cloned)))
        cloned = self.relu(self.bn3(self.conv3(cloned)))
        return x + cloned


class ConvLSTMCell(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size=3,
        bias=True,
    ):
        """
        Initializes ConvLSTM Cell

        Args:
            in_channels (int): Number of channels of input to cell
            hidden_channels (int): Number of channels of hidden state
            cell_state_height (int): Height of cell state
            cell_state_width (int): Width of cell state
            in_kernel (int, optional): Size of input->state kernel. Defaults to 3.
            hidden_kernel (int, optional): Size of state->state kernel. Defaults to 3.
            bias (bool, optional): Enables biases on convolution layers. Defaults to True.
        """
        super(ConvLSTMCell, self).__init__()
        # Remember that convolutions are linear - combining all conv operations into a single combined
        # layer is no different that having 8 different convolution layers.
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=bias,
            dtype=torch.float32,
        )
        self.bn = torch.nn.BatchNorm2d(in_channels + hidden_channels)
        self.init_weights()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.hidden_channels = hidden_channels

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.conv.weight)
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, state=None):
        """
        Forward pass of ConvLSTM cell

        Args:
            x (torch.tensor): A (N,T,H,W) input tensor, with
            - N: batch dimension
            - T: time dimension on slices
            - H: height of inputs
            - W: width of inputs
            state (Tuple(torch.tensor), optional): Hidden and cell state tensors. Defaults to None.

        Returns:
            _type_: _description_
        """
        h, c = state
        stack = self.bn(torch.cat((x, h), dim=1))
        f, i, cell, o = torch.split(self.conv(stack), self.hidden_channels, dim=1)
        ct = f * c + i * self.tanh(cell)
        ht = o * self.tanh(ct)
        return ht, ct
