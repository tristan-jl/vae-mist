from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def calculate_conv_output_size(input, padding, kernel_size, stride) -> int:
    return int(((input + 2 * padding - 1 * (kernel_size - 1) - 1) / stride) + 1)


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        relu_slope: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = 1
        self.relu_slope = relu_slope

        conv_layer = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
        )

        leaky_relu = torch.nn.LeakyReLU(self.relu_slope)
        batch_norm = torch.nn.BatchNorm2d(self.out_channels)

        self.block = torch.nn.Sequential(conv_layer, leaky_relu, batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvTransposeBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        relu_slope: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = 1
        self.output_padding = self.stride - 1
        self.relu_slope = relu_slope

        conv_layer = torch.nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
        )

        leaky_relu = torch.nn.LeakyReLU(self.relu_slope)
        batch_norm = torch.nn.BatchNorm2d(self.out_channels)

        self.block = torch.nn.Sequential(conv_layer, leaky_relu, batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InBottleneck(torch.nn.Module):
    def __init__(self, in_features: int, latent_space_dim: int):
        super().__init__()

        self.mu = torch.nn.Linear(in_features, latent_space_dim)
        self.log_variance = torch.nn.Linear(in_features, latent_space_dim)

    @staticmethod
    def reparameterise(mu: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5 * log_variance)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.reshape(x.size(0), -1)  # flatten non-minibatch dims
        mu = self.mu(x)
        log_variance = self.log_variance(x)

        return mu, log_variance, self.reparameterise(mu, log_variance)


class OutBottleneck(torch.nn.Module):
    def __init__(self, latent_space_dim: int, unflatten_shape: tuple[int, int, int]):
        super().__init__()

        self.linear = torch.nn.Linear(
            latent_space_dim,
            np.prod(unflatten_shape),
        )
        self.unflatten_shape = unflatten_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = x.reshape(x.size(0), *self.unflatten_shape)  # unflatten non-minibatch dims
        return x


class VariationalAutoencoder(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_width: int,
        input_height: int,
        conv_out_channels: Sequence[int],
        conv_kernels: Sequence[int],
        conv_strides: Sequence[int],
        latent_space_dim: int,
    ):
        super().__init__()
        assert len(conv_out_channels) == len(conv_kernels) == len(conv_strides)

        self.input_width = input_width
        self.input_height = input_height

        self.conv_in_channels = [input_channels] + list(conv_out_channels)[:-1]
        self.conv_out_channels = list(conv_out_channels)
        self.conv_kernels = list(conv_kernels)
        self.conv_strides = list(conv_strides)
        self.latent_space_dim = latent_space_dim

        self.conv_shapes = self.__calculate_conv_shapes()

        self.encoder = self.__build_encoder()
        self.decoder = self.__build_decoder()

    def __calculate_conv_shapes(self):
        conv_shapes = [(self.conv_in_channels[0], self.input_width, self.input_height)]

        for o, k, s in zip(
            self.conv_out_channels,
            self.conv_kernels,
            self.conv_strides,
        ):
            output_width = calculate_conv_output_size(conv_shapes[-1][1], 1, k, s)
            output_height = calculate_conv_output_size(conv_shapes[-1][2], 1, k, s)
            conv_shapes.append((o, output_width, output_height))

        return conv_shapes

    def __build_encoder(self) -> torch.nn.Module:
        conv_layers = torch.nn.Sequential(
            *[
                ConvBlock(*args)
                for args in zip(
                    self.conv_in_channels,
                    self.conv_out_channels,
                    self.conv_kernels,
                    self.conv_strides,
                )
            ]
        )
        bottleneck = InBottleneck(np.prod(self.conv_shapes[-1]), self.latent_space_dim)

        return torch.nn.Sequential(conv_layers, bottleneck)

    def __build_decoder(self) -> torch.nn.Module:
        bottleneck = OutBottleneck(self.latent_space_dim, self.conv_shapes[-1])
        conv_layers = torch.nn.Sequential(
            *reversed(
                [
                    ConvTransposeBlock(*args)
                    for args in zip(
                        self.conv_out_channels,
                        self.conv_in_channels,
                        self.conv_kernels,
                        self.conv_strides,
                    )
                ]
            )
        )

        return torch.nn.Sequential(bottleneck, conv_layers, torch.nn.Sigmoid())

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_variance, representation = self.encoder(x)
        reconstruction = self.decoder(representation)
        return mu, log_variance, reconstruction


class VAELoss(torch.nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()

        self.alpha = alpha
        self.bce_loss_function = torch.nn.BCELoss(reduction="sum")

    @staticmethod
    def kl_loss_function(mu: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        mu, log_variance, input_x = inputs

        recon_loss = self.bce_loss_function(input_x, targets)
        kl_loss = self.kl_loss_function(mu, log_variance)

        return self.alpha * recon_loss + kl_loss
