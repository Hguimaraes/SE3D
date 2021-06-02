import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb


class FCNConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels:int=None, 
        kernel_size:int=55, 
        base_channels:int=80,
        use_sinc:bool=False
    ):
        super(FCNConvBlock, self).__init__()
        if in_channels is None:
            in_channels = base_channels

        self.conv_layer = sb.nnet.CNN.Conv1d
        if use_sinc:
            self.conv_layer = SincConv

        self.conv = nn.Sequential(
            self.conv_layer(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=kernel_size
            ),
            sb.nnet.normalization.InstanceNorm1d(
                input_size=base_channels,
                track_running_stats=False,
                affine=True
            ),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

"""
From paper: "End-to-End Waveform Utterance Enhancement for Direct Evaluation
Metrics Optimization by Fully Convolutional Neural Networks", TASLP, 2018
"""
class FCN(nn.Module):
    def __init__(
        self, 
        in_channels:int=8,
        out_channels:int=1,
        kernel_size:int=55,
        base_channels:int=80,
        use_sinc:bool=False
    ):
        super(FCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.base_channels = base_channels
        self.use_sinc = use_sinc

        # Construct FCN model as sequential
        self.model = nn.Sequential(
            # Instance norm 0
            sb.nnet.normalization.InstanceNorm1d(
                input_size=in_channels,
                track_running_stats=False,
                affine=True
            ),

            # Conv blocks
            FCNConvBlock(self.in_channels, self.kernel_size, self.base_channels, self.use_sinc),
            FCNConvBlock(self.base_channels, self.kernel_size, self.base_channels, False),
            FCNConvBlock(self.base_channels, self.kernel_size, self.base_channels, False),
            FCNConvBlock(self.base_channels, self.kernel_size, self.base_channels, False),
            FCNConvBlock(self.base_channels, self.kernel_size, self.base_channels, False),
            FCNConvBlock(self.base_channels, self.kernel_size, self.base_channels, False),
            FCNConvBlock(self.base_channels, self.kernel_size, self.base_channels, False),

            # Out conv
            sb.nnet.CNN.Conv1d(
                in_channels=self.base_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size
            )
        )
    
    def forward(self, x):
        x = x.transpose(1, 2) # B, C, L => B, L, C
        x = self.model(x)
        return x.transpose(1, 2) # B, L, C => B, C, L


# Adapted from speechbrain
# We apply the same filters across all channels in the first layer
class SincConv(nn.Module):
    """
    This function implements SincConv (SincNet).
    M. Ravanelli, Y. Bengio, "Speaker Recognition from raw waveform with
    SincNet", in Proc. of  SLT 2018 (https://arxiv.org/abs/1808.00158)
    """
    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding="same",
        padding_mode="reflect",
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # input shape inference
        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        # Initialize Sinc filters
        self._init_sinc_conv()

    def forward(self, x):
        """Returns the output of the convolution.
        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.
        """
        x = x.transpose(1, -1)
        self.device = x.device

        unsqueeze = x.ndim == 2
        if unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got %s."
                % (self.padding)
            )

        sinc_filters = self._get_sinc_filters()
        sinc_filters = sinc_filters.expand(sinc_filters.shape[0], self.in_channels, sinc_filters.shape[2])

        wx = F.conv1d(
            x,
            sinc_filters,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )

        if unsqueeze:
            wx = wx.squeeze(1)

        wx = wx.transpose(1, -1)

        return wx

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels.
        """

        if len(shape) == 2:
            in_channels = 1
        elif len(shape) == 3:
            in_channels = 1
        else:
            raise ValueError(
                "sincconv expects 2d or 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )
        return in_channels

    def _get_sinc_filters(self,):
        """This functions creates the sinc-filters to used for sinc-conv.
        """
        # Computing the low frequencies of the filters
        low = self.min_low_hz + torch.abs(self.low_hz_)

        # Setting minimum band and minimum freq
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        # Passing from n_ to the corresponding f_times_t domain
        self.n_ = self.n_.to(self.device)
        self.window_ = self.window_.to(self.device)
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Left part of the filters.
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low))
            / (self.n_ / 2)
        ) * self.window_

        # Central element of the filter
        band_pass_center = 2 * band.view(-1, 1)

        # Right part of the filter (sinc filters are symmetric)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        # Combining left, central, and right part of the filter
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        # Amplitude normalization
        band_pass = band_pass / (2 * band[:, None])

        # Setting up the filter coefficients
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return filters

    def _init_sinc_conv(self):
        """Initializes the parameters of the sinc_conv layer."""

        # Initialize filterbanks such that they are equally spaced in Mel scale
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = torch.linspace(
            self._to_mel(self.min_low_hz),
            self._to_mel(high_hz),
            self.out_channels + 1,
        )

        hz = self._to_hz(mel)

        # Filter lower frequency and bands
        self.low_hz_ = hz[:-1].unsqueeze(1)
        self.band_hz_ = (hz[1:] - hz[:-1]).unsqueeze(1)

        # Maiking freq and bands learnable
        self.low_hz_ = nn.Parameter(self.low_hz_)
        self.band_hz_ = nn.Parameter(self.band_hz_)

        # Hamming window
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )
        self.window_ = 0.54 - 0.46 * torch.cos(
            2 * math.pi * n_lin / self.kernel_size
        )

        # Time axis  (only half is needed due to symmetry)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )

    def _to_mel(self, hz):
        """Converts frequency in Hz to the mel scale.
        """
        return 2595 * np.log10(1 + hz / 700)

    def _to_hz(self, mel):
        """Converts frequency in the mel scale to Hz.
        """
        return 700 * (10 ** (mel / 2595) - 1)

    def _manage_padding(
        self, x, kernel_size: int, dilation: int, stride: int,
    ):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.
        """

        # Detecting input shape
        L_in = x.shape[-1]

        # Time padding
        padding = sb.nnet.CNN.get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x