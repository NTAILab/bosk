from operator import mul
from joblib import Parallel, delayed
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from sklearn.utils.multiclass import check_classification_targets
from typing import List, Optional, Tuple, Union
from functools import partial, reduce

from ....base import BaseBlock, TransformOutputData, BlockInputData
from ....meta import BlockMeta, BlockExecutionProperties, InputSlotMeta, OutputSlotMeta
from .....stages import Stages
from .....data import CPUData, GPUData
from .....utility import get_random_generator, get_rand_int
from ...multi_grained_scanning._convolution_helpers import _ConvolutionParams, _ConvolutionHelper
from .ferns import calculate_bucket_stats, predict_proba


__all__ = [
    "MGSRandomFernsBlock",
]


@partial(jax.jit, static_argnames=('n_channels', 'window_size', 'n_ferns', 'fern_size'))
def make_window_unary_ferns(xs: jnp.ndarray,
                            n_channels: int,
                            window_size: int,
                            n_ferns: int,
                            fern_size: int,
                            key: random.KeyArray):
    """Generate indices and threshold values for unary fern that is applied to a sliding window.
    The sliding window captures all channels simultaneously.

    Args:
        xs: Input data of shape (n_samples, n_channels, n_1, ..., n_k).
        n_channels: Number of input data channels.
        window_size: Flattened window size.
        n_ferns: Number of ferns to generate.
        fern_size: Size of fern (statistics array size is exponential to the size of fern).
        key: PRNG Key.

    Returns:
        Pair (feature indices, feature thresholds), both of shape (n_ferns, fern_size).

    """
    total_window_size = n_channels * window_size
    ft_key, fi_key = random.split(key)
    n_indices = n_ferns * fern_size
    feature_indices = random.randint(fi_key, (n_indices,), minval=0, maxval=total_window_size)
    mins = xs.min()
    maxs = xs.max()
    feature_thresholds = random.uniform(
        ft_key,
        (n_indices,),
        minval=mins,
        maxval=maxs,
    )
    return feature_indices.reshape((n_ferns, fern_size)), feature_thresholds.reshape((n_ferns, fern_size))


@partial(jax.jit, static_argnames=('n_channels', 'n_corners', 'n_kernel_points', 'pooled_shape'))
def apply_window_unary_ferns(xs: jnp.ndarray, raveled_pooling_indices,
                             n_channels, n_corners, n_kernel_points,
                             pooled_shape,
                             feature_indices, feature_thresholds):
    """Compute the bucket indices for each sliding window.

    Args:
        xs: Input data of shape (n_samples, n_channels * n_1 * ... * n_k).
        raveled_pooling_indices: Raveled indices of elements of each window.
        n_channels: Number of channels.
        n_corners: Number of corners.
        n_kernel_points: Number of points in kernel.
        pooled_shape: Pooled shape.
        feature_indices: Ferns feature indices of shape (n_ferns, fern_size).
        feature_thresholds: Ferns thresholds of shape (n_ferns, fern_size).

    Returns:
        Bucket indices of shape (n_samples, n_1, ..., n_k, n_ferns).

    """
    n_ferns = feature_indices.shape[0]
    fern_size = feature_indices.shape[1]
    index_multipliers = 2 ** jnp.arange(fern_size)

    result = jax.vmap(
        lambda x: (
            (
                x[raveled_pooling_indices].reshape((
                    n_channels, n_corners, n_kernel_points
                )).swapaxes(0, 1).reshape((n_corners, -1))[:, feature_indices] >= feature_thresholds[jnp.newaxis]
            ) @ index_multipliers
        ).reshape((*pooled_shape, n_ferns)),
        in_axes=0,
        out_axes=0
    )(xs)
    return result


@partial(jax.jit, static_argnames=('n_channels', 'window_size', 'n_ferns', 'fern_size'))
def make_window_binary_ferns(n_channels: int,
                             window_size: int,
                             n_ferns: int,
                             fern_size: int,
                             key: random.KeyArray):
    """Generate pair indices for binary fern which is applied to a sliding window.
    The sliding window captures all channels simultaneously.

    Args:
        xs: Input data of shape (n_samples, n_channels, n_1, ..., n_k).
        n_channels: Number of input data channels.
        window_size: Flattened window size.
        n_ferns: Number of ferns to generate.
        fern_size: Size of fern (statistics array size is exponential to the size of fern).
        key: PRNG Key.

    Returns:
        Pair (feature indices left, feature indices right).

    """
    total_window_size = n_channels * window_size
    left_key, right_key = random.split(key)
    n_indices = n_ferns * fern_size
    left_indices = random.randint(left_key, (n_indices,), minval=0, maxval=total_window_size)
    right_indices = random.randint(right_key, (n_indices,), minval=0, maxval=total_window_size)
    return left_indices.reshape((n_ferns, fern_size)), right_indices.reshape((n_ferns, fern_size))


@partial(jax.jit, static_argnames=('n_channels', 'n_corners', 'n_kernel_points', 'pooled_shape'))
def apply_window_binary_ferns(xs: jnp.ndarray, raveled_pooling_indices,
                              n_channels, n_corners, n_kernel_points,
                              pooled_shape,
                              left_indices, right_indices):
    """Compute the bucket indices for each sliding window.

    Args:
        xs: Input data of shape (n_samples, n_channels * n_1 * ... * n_k).
        raveled_pooling_indices: Raveled indices of elements of each window.
        n_channels: Number of channels.
        n_corners: Number of corners.
        n_kernel_points: Number of points in kernel.
        pooled_shape: Pooled shape.
        left_indices: Ferns left feature indices of shape (n_ferns, fern_size).
        right_indices: Ferns right feature indices of shape (n_ferns, fern_size).

    Returns:
        Bucket indices of shape (n_samples, n_ferns).

    """
    n_ferns = left_indices.shape[0]
    fern_size = left_indices.shape[1]
    index_multipliers = 2 ** jnp.arange(fern_size)

    def compare(x):
        return x[:, left_indices] >= x[:, right_indices]

    result = jax.vmap(
        lambda x: (
            compare(
                x[raveled_pooling_indices].reshape((
                    n_channels, n_corners, n_kernel_points
                )).swapaxes(0, 1).reshape((n_corners, -1))
            ) @ index_multipliers
        ).reshape((*pooled_shape, n_ferns)),
        in_axes=0,
        out_axes=0
    )(xs)
    return result


class MGSRandomFernsBlock(BaseBlock):
    """Multi-Grained Scanning Random Ferns Classifier Block.
    It applies multiple tests to each sample across spatial dimensions.
    The result is of the same shape as pooling result (with the same convolution parameters).

    The implementation is based on idea that each patch can be considered as a separate training sample.

    Args:
        n_groups: Number of ferns groups (like a number of estimators).
        n_ferns_in_group: Number of ferns in a group.
        fern_size: Number of tests in fern.
        kind: Kind of tests ('unary' or 'binary').
        bootstrap: Apply data bootstrap or not.
        n_jobs: Number of threads.
        random_state: Random state.

        kernel_size: Kernel size (int or tuple).
        stride: Stride.
        dilation: Dilation (kernel stride).
        padding: Padding size (see `numpy.pad`);
                    if None padding is disabled.


    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        - X: Input features.
        - y: Ground truth labels.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - X: Input features.

    Output slots
    ------------

        - probas: Predicted probabilities.

    """
    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='X',
                stages=Stages(transform=True),
            ),
            InputSlotMeta(
                name='y',
                stages=Stages(transform=False, transform_on_fit=True),
            )
        ],
        outputs=[
            OutputSlotMeta(
                name='probas',
            )
        ],
        execution_props=BlockExecutionProperties(gpu=True),
    )
    pooling_indices_ = None

    def __init__(self, n_groups: int = 10,
                 n_ferns_in_group: int = 20,
                 fern_size: int = 7,
                 kind: str = 'unary',
                 bootstrap: bool = False,
                 n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[None, int, Tuple[int]] = None,
                 dilation: int = 1,
                 padding: Optional[int] = None):
        super().__init__()
        assert fern_size <= 32, 'Maximum number of tests in a fern is 32'
        self.n_groups = n_groups
        self.n_ferns_in_group = n_ferns_in_group
        self.n_ferns = n_ferns_in_group * n_groups
        self.fern_size = fern_size
        self.kind = kind
        self.n_buckets = 2 ** fern_size
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.params = _ConvolutionParams(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        self.helper_ = _ConvolutionHelper(self.params)
        self.pooling_indices_ = None

    def __getstate__(self) -> dict:
        ATTRS = (
            'n_groups', 'n_ferns_in_group', 'n_ferns', 'fern_size',
            'kind', 'n_buckets', 'bootstrap', 'n_jobs', 'random_state',
            'n_classes_', 'params', 'helper_',
            'slots',  # BaseBlock attribute
        )
        state = {
            k: getattr(self, k)
            for k in ATTRS
            if hasattr(self, k)
        }
        TO_NUMPY = ('ferns_', 'prior_', 'bucket_stats_', 'classes_')
        for k in TO_NUMPY:
            if hasattr(self, k):
                v = getattr(self, k)
                if k == 'ferns_':
                    # ferns is an exception, the tuple may contain arrays of different types
                    state[k] = tuple(np.asarray(el) for el in v)
                else:
                    state[k] = np.asarray(getattr(self, k))
        return state

    def __setstate__(self, state: dict):
        for k, v in state.items():
            if k == 'ferns_':
                v = tuple(jnp.asarray(el) for el in v)
            elif isinstance(v, np.ndarray):
                v = jnp.asarray(v)
            setattr(self, k, v)

    def _classifier_init(self, y):
        check_classification_targets(y)

        self.classes_, y_encoded = jnp.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        y = y_encoded
        return y

    def _get_flattened_window_size(self, X):
        return reduce(
            mul,
            self.helper_.check_kernel_size(n_spatial_dims=(X.ndim - 2)),
            1
        )

    def _make_ferns(self, X, prng_key):
        n_channels = X.shape[1]
        if self.kind == 'unary':
            return make_window_unary_ferns(
                X,
                n_channels=n_channels,
                window_size=self._get_flattened_window_size(X),
                n_ferns=self.n_ferns,
                fern_size=self.fern_size,
                key=prng_key
            )
        elif self.kind == 'binary':
            return make_window_binary_ferns(
                n_channels=n_channels,
                window_size=self._get_flattened_window_size(X),
                n_ferns=self.n_ferns,
                fern_size=self.fern_size,
                key=prng_key
            )
        else:
            raise ValueError(f'Wrong kind: {self.kind!r}')

    def _apply_ferns(self, xs, ferns):
        if self.params.padding is not None:
            xs = self.helper_.pad(xs)
        pooling_indices = self.__prepare_pooling_indices(xs.shape)
        n_channels = xs.shape[1]
        raveled_xs = xs.reshape((xs.shape[0], -1))
        raveled_pooling_indices = jnp.ravel_multi_index(pooling_indices.full_index_tuple, xs.shape[1:])
        if self.kind == 'unary':
            return apply_window_unary_ferns(
                raveled_xs,
                raveled_pooling_indices,
                n_channels,
                pooling_indices.n_corners,
                pooling_indices.n_kernel_points,
                pooling_indices.pooled_shape,
                *ferns
            )
        elif self.kind == 'binary':
            return apply_window_binary_ferns(
                raveled_xs,
                raveled_pooling_indices,
                n_channels,
                pooling_indices.n_corners,
                pooling_indices.n_kernel_points,
                pooling_indices.pooled_shape,
                *ferns
            )
        else:
            raise ValueError(f'Wrong kind: {self.kind!r}')

    def __prepare_pooling_indices(self, xs_shape):
        if self.pooling_indices_ is not None and xs_shape == self.pooling_indices_.xs_shape:
            return self.pooling_indices_

        self.pooling_indices_ = self.helper_.prepare_pooling_indices(xs_shape)
        return self.pooling_indices_

    def _parallel_calc_bucket_stats(self, bucket_indices, y, group_data_indices):
        bucket_stats = Parallel(n_jobs=self.n_jobs, prefer='threads')(
            delayed(calculate_bucket_stats)(
                bucket_indices[
                    data_ids,
                    slice(i * self.n_ferns_in_group, (i + 1) * self.n_ferns_in_group)
                ],
                self.n_buckets,
                y[data_ids],
                n_classes=self.n_classes_,
            )
            for i, data_ids in enumerate(group_data_indices)
        )
        bucket_stats = jnp.concatenate(bucket_stats, axis=1)
        return bucket_stats

    def fit(self, inputs: BlockInputData) -> 'MGSRandomFernsBlock':
        """Fit the MGS Random Ferns Block.
        The implementation is device-agnostic.

        """
        assert type(inputs['X']) == type(inputs['y'])  # noqa: E721
        X = inputs['X'].data
        y = inputs['y'].data
        y = self._classifier_init(y)
        np_rng = get_random_generator(self.random_state)
        prng_key = random.PRNGKey(get_rand_int(np_rng))
        prng_key, ferns_key = random.split(prng_key)
        n_samples = X.shape[0]

        ferns = self._make_ferns(X, ferns_key)
        bucket_indices = self._apply_ferns(X, ferns)
        # bucket_indices shape: (n_samples, n_1, ..., n_k, n_ferns)
        n_ferns = bucket_indices.shape[-1]
        flattened_bucket_indices = bucket_indices.reshape((-1, n_ferns))
        spatial_size = reduce(mul, bucket_indices.shape[1:-1], 1)
        flattened_y = jnp.tile(y[:, np.newaxis], (1, spatial_size)).reshape((-1, *y.shape[1:]))

        group_data_indices: List[jnp.ndarray] | List[slice]
        if not self.bootstrap:
            group_data_indices = [slice(None, None) for _ in range(self.n_groups)]
        else:
            keys = random.split(prng_key, self.n_groups)
            group_data_indices = [
                random.choice(keys[i], n_samples, shape=(n_samples,), replace=True)
                for i in range(self.n_groups)
            ]

        if self.n_jobs is None and not self.bootstrap:
            bucket_stats = calculate_bucket_stats(
                flattened_bucket_indices,
                self.n_buckets,
                flattened_y,
                n_classes=self.n_classes_
            )
        else:
            bucket_stats = self._parallel_calc_bucket_stats(
                flattened_bucket_indices,
                flattened_y,
                group_data_indices
            )

        _classes, counts = jnp.unique(y, return_counts=True)
        self.prior_ = counts / counts.sum()
        self.ferns_ = ferns
        self.bucket_stats_ = bucket_stats
        return self

    def _predict_proba(self, X):
        group_preds = []
        for group_indices in jnp.split(jnp.arange(self.n_ferns), self.n_groups):
            prediction_bucket_indices = self._apply_ferns(X, [f[group_indices] for f in self.ferns_])
            n_ferns = prediction_bucket_indices.shape[-1]
            flattened_bucket_indices = prediction_bucket_indices.reshape((-1, n_ferns))
            spatial_dims = prediction_bucket_indices.shape[1:-1]
            probas = predict_proba(
                flattened_bucket_indices,
                self.bucket_stats_[:, group_indices],
                prior=self.prior_
            )
            group_preds.append(
                probas.reshape((-1, *spatial_dims, probas.shape[-1]))
            )
        preds = jnp.moveaxis(jnp.stack(group_preds, axis=0).mean(axis=0), -1, 1)
        return preds

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        X = inputs['X'].data

        result = self._predict_proba(X)
        if type(inputs['X']) == CPUData:
            return {'probas': CPUData(np.asarray(result))}
        elif type(inputs['X']) == GPUData:
            return {'probas': GPUData(result)}
        else:
            raise NotImplementedError(f'Not implemented for input type: {type(inputs["X"])!r}')
