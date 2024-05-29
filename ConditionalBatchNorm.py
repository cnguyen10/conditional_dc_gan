import jax
from jax import numpy as jnp
from jax.nn import initializers

import flax.linen as nn
from flax.linen import dtypes, module
from flax.linen.normalization import _canonicalize_axes, _compute_stats

from flax.typing import (
  Array,
  PRNGKey as PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  Axes,
)
from typing import Optional, Any


compact = module.compact


def _normalize(
    mdl: nn.Module,
    x: Array,
    cls_indices: Array | list[int],
    num_classes: int,
    mean: Array,
    var: Array,
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: Optional[Dtype],
    param_dtype: Dtype,
    epsilon: float,
    use_bias: bool,
    use_scale: bool,
    bias_init: Initializer,
    scale_init: Initializer,
):
    """Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

    Arguments:
    mdl: Module to apply the normalization in (normalization params will reside
        in this module).
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
        for each specified feature.
    dtype: The dtype of the result (default: infer from input and params).
    param_dtype: The dtype of the parameters.
    epsilon: Normalization epsilon.
    use_bias: If true, add a bias term to the output.
    use_scale: If true, scale the output.
    bias_init: Initialization function for the bias term.
    scale_init: Initialization function for the scaling function.

    Returns:
    The normalized input.
    """
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])

    mean = jnp.expand_dims(mean, reduction_axes)
    var = jnp.expand_dims(var, reduction_axes)
    y = x - mean
    mul = jax.lax.rsqrt(var + epsilon)
    args = [x]
    if use_scale:
        scale = mdl.param(
            'scale',
            scale_init,
            [num_classes,] + reduced_feature_shape,
            # reduced_feature_shape,
            param_dtype
        ).reshape([num_classes,] + feature_shape[1:])
        # ).reshape(feature_shape)
        mul *= scale[cls_indices]
        # mul *= scale
        args.append(scale)
    y *= mul
    if use_bias:
        bias = mdl.param(
            'bias',
            bias_init,
            [num_classes,] + reduced_feature_shape,
            # reduced_feature_shape,
            param_dtype
        ).reshape([num_classes,] + feature_shape[1:])
        # ).reshape(feature_shape)
        y += bias[cls_indices]
        # y += bias
        args.append(bias)
    dtype = dtypes.canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, dtype)


class ConditionalBatchNorm(nn.Module):
    num_classes: int  # new attribute for the conditioning
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Initializer = initializers.zeros
    scale_init: Initializer = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True
    force_float32_reductions: bool = True

    @compact
    def __call__(
        self,
        x: jax.typing.ArrayLike,
        cls_indices: jax.typing.ArrayLike | list[int],
        use_running_average: Optional[bool] = None,
        *,
        mask: Optional[jax.Array] = None,
    ):
        """Normalizes the input using batch statistics.

        .. note::
            During initialization (when ``self.is_initializing()`` is ``True``) the running
            average of the batch statistics will not be updated. Therefore, the inputs
            fed during initialization don't need to match that of the actual input
            distribution and the reduction axis (set with ``axis_name``) does not have
            to exist.

        Args:
            x: the input to be normalized.
            y: the class indices of x.
            use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.
            mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.

        Returns:
            Normalized inputs (the same shape as inputs).
        """

        use_running_average = module.merge_param(
            'use_running_average', self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            'batch_stats',
            'mean',
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
                mask=mask,
                force_float32_reductions=self.force_float32_reductions,
            )

            if not self.is_initializing():
                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        return _normalize(
            self,
            x,
            cls_indices,
            self.num_classes,
            mean,
            var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )
