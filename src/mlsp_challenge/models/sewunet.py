import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def reserve_pop(x):
    # Helper function to reverse a list
    return x[::-1][:-1]

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int,
        activation:object,
        padding:int=None, 
        dilation:int=1, 
        kernel_size:int=15,
        stride:int=1
    ):
        """"""
        super(ConvBlock, self).__init__()
        self.padding = padding or (kernel_size // 2)
        self.padding_layer = nn.ReflectionPad1d(self.padding)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation
        )
        self.batch = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.padding_layer(x)
        x = self.conv(x)
        x = self.batch(x)
        x = self.activation(x)

        return x


"""
Encoder block of the Fully-convolutional Network
"""
class DownSamplingBlock(nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int,
        activation:object,
        padding:int=None, 
        dilation:int=1, 
        kernel_size:int=15
    ):
        super(DownSamplingBlock, self).__init__()
        self.block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

    def forward(self, x):
        x = self.block(x)
        return x[:, :, ::2], x


"""
Decoder block of the Fully-convolutional Network
"""
class UpSamplingBlock(nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int,
        activation:object,
        padding:int=None, 
        dilation:int=1, 
        kernel_size:int=15,
        mode="linear"
    ):
        super(UpSamplingBlock, self).__init__()
        self.mode = mode

        # Convolution block
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        # Deconvolution block
        if not self.mode == "linear":
            self.deconv = nn.ConvTranspose1d(
                in_channels=in_channels - out_channels,
                out_channels=in_channels - out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=True,
                dilation=1
            )
        else:
            self.deconv = lambda x: F.interpolate(
                x, 
                scale_factor=2,
                mode='linear', 
                align_corners=True
            )

            # self.deconv_activation = activation

    def forward(self, x, x_enc):
        x = self.deconv(x)
        x = torch.cat([x, x_enc], dim=1) # Concat with Skip connection
        return self.conv(x)


"""
    Convolutional block similar to VSConvBlock.
    The network input is fed into this layer
"""
class OutBlock(nn.Module):
    def __init__(
        self,
        in_channels:int, 
        out_channels:int,
        activation:object,
        padding:int=None, 
        dilation:int=1, 
        kernel_size:int=15,
    ):
        super(OutBlock, self).__init__()
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

    def forward(self, x, x_enc):
        x = torch.cat([x, x_enc], dim=1)
        return self.conv(x) - x_enc.mean(dim=1).unsqueeze(1)


"""
"""
class SEWUNet(nn.Module):
    def __init__(
        self,
        in_channels:int=8,
        out_channels:int=1,
        depth:int=5,
        fsize:int=15,
        moffset:int=10,
        fd:int=15, 
        fu:int=5
    ):
        """Speech Enhancenment using Wave-U-Net"""
        super(SEWUNet, self).__init__()

        # Hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.fsize = fsize
        self.moffset = moffset
        self.fd = fd
        self.fu = fu

        # Generate the list of in, out channels for the encoder
        self.enc_filters = [self.in_channels]
        self.enc_filters += [self.fsize * i + self.moffset
                             for i in range(1, self.depth + 1)]
        self.n_encoder = zip(self.enc_filters, self.enc_filters[1:])

        # Bottleneck block sizes
        mid_in = self.fsize * self.depth + self.moffset
        mid_out = self.fsize * (self.depth + 1) + self.moffset

        # Generate the list of in, out channels for the decoder
        self.out_dec = reserve_pop(self.enc_filters)
        self.in_dec = [mid_out + self.enc_filters[-1]]
        self.in_dec += [self.out_dec[i] + self.out_dec[i + 1]
                        for i in range(self.depth - 1)]
        self.n_decoder = zip(self.in_dec, self.out_dec)

        # Architecture and parameters
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Build the encoder part of the U-net architecture
        for i, (in_ch, out_ch) in enumerate(self.n_encoder):
            self.encoder.append(
                DownSamplingBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=self.fd,
                    padding=self.fd // 2,
                    activation=nn.LeakyReLU(0.1)
                )
            )

        # Bottleneck block for the U-net
        self.mid_block = ConvBlock(
            in_channels=mid_in,
            out_channels=mid_out,
            kernel_size=self.fd,
            padding=self.fd // 2,
            activation=nn.LeakyReLU(0.1)
        )

        # Build the decoder part of the U-net architecture
        for in_ch, out_ch in self.n_decoder:
            self.decoder.append(
                UpSamplingBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=self.fu,
                    padding=self.fu // 2,
                    activation=nn.LeakyReLU(0.1)
                )
            )

        # Output block
        out_ch = self.out_dec[-1] + self.in_channels
        self.out_block = OutBlock(
            in_channels=out_ch,
            out_channels=self.out_channels,
            activation=nn.Tanh()
        )

    def forward(self, x):
        """"""
        enc = []
        net_in = copy.copy(x)

        # Encoder
        for i in range(self.depth):
            x, xi = self.encoder[i](x)
            enc.append(xi)

        x = self.mid_block(x)

        # Decoder
        for i in range(self.depth):
            x = self.decoder[i](x, enc.pop())

        x = self.out_block(x, net_in)

        return x