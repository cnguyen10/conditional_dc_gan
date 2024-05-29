import jax
from flax import linen as nn

from typing import Optional
import chex

from ConditionalBatchNorm import ConditionalBatchNorm


class Generator(nn.Module):
    ngf: int  # number of transpose convolutional channels
    nc: int  # number of channels in input samples
    train: Optional[bool] = None

    @nn.compact
    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> chex.Array:
        train = nn.merge_param(name='train', a=self.train, b=train)

        out = nn.ConvTranspose(
            features=self.ngf * 8,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding='VALID',
            use_bias=False
        )(inputs=x)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        out = nn.ConvTranspose(
            features=self.ngf * 4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            use_bias=False
        )(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        out = nn.ConvTranspose(
            features=self.ngf * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            use_bias=False
        )(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        out = nn.ConvTranspose(
            features=self.ngf,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            use_bias=False
        )(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        out = nn.ConvTranspose(
            features=self.nc,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            use_bias=False
        )(inputs=out)
        out = nn.sigmoid(out)

        return out


class Discriminator(nn.Module):
    ndf: int  # number of convolutional channels
    train: Optional[bool] = None

    @nn.compact
    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> chex.Array:
        train = nn.merge_param(name='train', a=self.train, b=train)

        out = nn.Conv(
            features=self.ndf,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=1,
            use_bias=False
        )(inputs=x)
        out = nn.leaky_relu(x=out, negative_slope=0.2)

        out = nn.Conv(
            features=self.ndf * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=1,
            use_bias=False
        )(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.leaky_relu(x=out, negative_slope=0.2)

        out = nn.Conv(
            features=self.ndf * 4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=1,
            use_bias=False
        )(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.leaky_relu(x=out, negative_slope=0.2)

        out = nn.Conv(
            features=self.ndf * 8,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=1,
            use_bias=False
        )(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.leaky_relu(x=out, negative_slope=0.2)

        out = nn.Conv(
            features=1,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding=0,
            use_bias=False
        )(inputs=out)

        out = jax.lax.collapse(operand=out, start_dimension=1).squeeze()

        return out


class ConditionalGenerator(nn.Module):
    ngf: int  # number of transpose convolutional channels
    nc: int  # number of channels in input samples
    num_classes: int
    train: Optional[bool] = None

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        y: chex.Array | list[int],
        train: Optional[bool] = None
    ) -> chex.Array:
        train = nn.merge_param(name='train', a=self.train, b=train)

        out = nn.ConvTranspose(
            features=self.ngf * 8,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding='VALID',
            use_bias=False
        )(inputs=x)
        out = ConditionalBatchNorm(num_classes=self.num_classes)(
            x=out,
            cls_indices=y,
            use_running_average=not train
        )
        out = nn.relu(x=out)

        out = nn.ConvTranspose(
            features=self.ngf * 4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            use_bias=False
        )(inputs=out)
        out = ConditionalBatchNorm(num_classes=self.num_classes)(
            x=out,
            cls_indices=y,
            use_running_average=not train
        )
        out = nn.relu(x=out)

        out = nn.ConvTranspose(
            features=self.ngf * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            use_bias=False
        )(inputs=out)
        out = ConditionalBatchNorm(num_classes=self.num_classes)(
            x=out,
            cls_indices=y,
            use_running_average=not train
        )
        out = nn.relu(x=out)

        out = nn.ConvTranspose(
            features=self.ngf,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            use_bias=False
        )(inputs=out)
        out = ConditionalBatchNorm(num_classes=self.num_classes)(
            x=out,
            cls_indices=y,
            use_running_average=not train
        )
        out = nn.relu(x=out)

        out = nn.ConvTranspose(
            features=self.nc,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            use_bias=False
        )(inputs=out)
        out = nn.sigmoid(out)

        return out