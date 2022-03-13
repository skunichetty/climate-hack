import torch


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
        self.inbn = torch.nn.BatchNorm2d(in_channels + hidden_channels)
        self.obn = torch.nn.BatchNorm2d(4 * hidden_channels)
        self.init_weights()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.hidden_channels = hidden_channels

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.conv.weight)
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, state):
        """
        Forward pass of ConvLSTM cell

        Args:
            x (torch.tensor): A (N,T,H,W) input tensor, with
            - N: batch dimension
            - T: time dimension on slices
            - H: height of inputs
            - W: width of inputs
            state (Tuple(torch.tensor)): Hidden and cell state tensors.

        Returns:
            Tuple(torch.tensor, torch.tensor): Output hidden and cell state tensors.
        """
        h, c = state
        stack = self.inbn(torch.cat((x, h), dim=1))
        f, i, cell, o = torch.split(
            self.obn(self.conv(stack)), self.hidden_channels, dim=1
        )
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        ct = f * c + i * self.tanh(cell)
        ht = o * self.tanh(ct)
        return ht, ct

    def init_hidden(self, batch_size, in_height, in_width):
        h = torch.zeros(
            (batch_size, self.hidden_channels, in_height, in_width),
            device=self.conv.weight.device,
            dtype=torch.float32,
        )
        c = torch.zeros(
            (batch_size, self.hidden_channels, in_height, in_width),
            device=self.conv.weight.device,
            dtype=torch.float32,
        )
        return h, c


class GroupedCLSTMCell(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size=3,
        num_groups=4,
        bias=True,
    ):
        super(GroupedCLSTMCell, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=bias,
            groups=num_groups,
            dtype=torch.float32,
        )
        self.obn = torch.nn.BatchNorm2d(4 * hidden_channels)
        self.init_weights()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.hidden_channels = hidden_channels

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.conv.weight)
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, state):
        """
        Forward pass of ConvLSTM cell

        Args:
            x (torch.tensor): A (N,T,H,W) input tensor, with
            - N: batch dimension
            - T: time dimension on slices
            - H: height of inputs
            - W: width of inputs
            state (Tuple(torch.tensor)): Hidden and cell state tensors.

        Returns:
            Tuple(torch.tensor, torch.tensor): Output hidden and cell state tensors.
        """
        h, c = state
        stack = torch.cat((x, h), dim=1)
        stack = self.obn(self.conv(stack))
        f, i, cell, o = torch.split(stack, self.hidden_channels, dim=1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        ct = f * c + i * self.tanh(cell)
        ht = o * self.tanh(ct)
        return ht, ct

    def init_hidden(self, batch_size, in_height, in_width):
        h = torch.zeros(
            (batch_size, self.hidden_channels, in_height, in_width),
            device=self.conv.weight.device,
            dtype=torch.float32,
        )
        c = torch.zeros(
            (batch_size, self.hidden_channels, in_height, in_width),
            device=self.conv.weight.device,
            dtype=torch.float32,
        )
        return h, c
