import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from sklearn.utils.multiclass import check_classification_targets
from joblib import Parallel, delayed
from typing import Optional


from ....base import BaseBlock, TransformOutputData, BlockInputData
from ....meta import BlockMeta, BlockExecutionProperties
from ....slot import InputSlotMeta, OutputSlotMeta
from .....stages import Stages
from .....data import CPUData, GPUData
from .....utility import get_random_generator, get_rand_int


@partial(jax.jit, static_argnames=('n_ferns', 'fern_size'))
def make_unary_ferns(xs: jnp.ndarray, n_ferns: int, fern_size: int, key: random.PRNGKey):
    """Generate indices and threshold values for unary fern.
    An unary fern represents a function that maps x to an integer number:
        x -> [ x[i[0]] >= t[0], ..., x[i[k]] >= t[k] ].

    Args:
        xs: Input data of shape (n_samples, n_features).
        n_ferns: Number of ferns to generate.
        fern_size: Size of fern (statistics array size is exponential to the size of fern).
        key: PRNG Key.

    Returns:
        Pair (feature indices, feature thresholds), both of shape (n_ferns, fern_size).

    """
    n_features = xs.shape[1]
    ft_key, fi_key = random.split(key)
    n_indices = n_ferns * fern_size
    feature_indices = random.randint(fi_key, (n_indices,), minval=0, maxval=n_features)
    mins = xs[:, feature_indices].min(axis=0)
    maxs = xs[:, feature_indices].max(axis=0)
    feature_thresholds = random.uniform(
        ft_key,
        (n_indices,),
        minval=mins,
        maxval=maxs,
    )
    return feature_indices.reshape((n_ferns, fern_size)), feature_thresholds.reshape((n_ferns, fern_size))


@jax.jit
def apply_unary_ferns(xs: jnp.ndarray, feature_indices, feature_thresholds):
    """Compute the bucket indices.

    Args:
        xs: Input data of shape (n_samples, n_features).
        feature_indices: Ferns feature indices of shape (n_ferns, fern_size).
        feature_thresholds: Ferns thresholds of shape (n_ferns, fern_size).

    Returns:
        Bucket indices of shape (n_samples, n_ferns).

    """
    fern_size = feature_indices.shape[1]
    index_multipliers = 2 ** jnp.arange(fern_size)
    predicates = (xs[:, feature_indices] >= feature_thresholds[jnp.newaxis])
    # predicates shape: (n_samples, n_ferns, fern_size)
    bucket_indices = predicates @ index_multipliers
    # bucket_indices shape: (n_samples, n_ferns)
    return bucket_indices


@partial(jax.jit, static_argnames=('n_ferns', 'fern_size'))
def make_binary_ferns(xs: jnp.ndarray, n_ferns: int, fern_size: int, key: random.PRNGKey):
    """Generate indices and threshold values for unary fern.
    An unary fern represents a function that maps x to an integer number:
        x -> [ x[i[0]] >= x[j[0]], ..., x[i[k]] >= x[j[k]] ].

    Args:
        xs: Input data of shape (n_samples, n_features).
        n_ferns: Number of ferns to generate.
        fern_size: Size of fern (statistics array size is exponential to the size of fern).
        key: PRNG Key.

    Returns:
        Pair (feature indices left, feature indices right).

    """
    n_features = xs.shape[1]
    left_key, right_key = random.split(key)
    n_indices = n_ferns * fern_size
    left_indices = random.randint(left_key, (n_indices,), minval=0, maxval=n_features)
    right_indices = random.randint(right_key, (n_indices,), minval=0, maxval=n_features)
    return left_indices.reshape((n_ferns, fern_size)), right_indices.reshape((n_ferns, fern_size))


@jax.jit
def apply_binary_ferns(xs: jnp.ndarray, left_indices, right_indices):
    """Compute the bucket indices.
    For each sample and fern bucket index indicates to which bucket the sample is assigned.

    Args:
        xs: Input data of shape (n_samples, n_features).
        left_indices: Ferns left feature indices of shape (n_ferns, fern_size).
        right_indices: Ferns right feature indices of shape (n_ferns, fern_size).

    Returns:
        Bucket indices of shape (n_samples, n_ferns).

    """
    fern_size = left_indices.shape[1]
    index_multipliers = 2 ** jnp.arange(fern_size)
    predicates = (xs[:, left_indices] >= xs[:, right_indices])
    # predicates shape: (n_samples, n_ferns, fern_size)
    bucket_indices = predicates @ index_multipliers
    # bucket_indices shape: (n_samples, n_ferns)
    return bucket_indices


@partial(jax.jit, static_argnames=('n_buckets', 'n_classes'))
def calculate_bucket_stats(bucket_indices: jnp.ndarray, n_buckets: int, y: jnp.ndarray, n_classes: int):
    """Calculate bucket statistics for each class.

    Args:
        bucket_indices: Bucket indices of shape (n_samples, n_ferns).
        n_buckets: Number of buckets (maximum value of `bucket_indices`).
        y: Target class values.
        n_classes: Number of classes.

    Returns:
        Bucket statistics of shape (n_classes, n_ferns, n_buckets).

    """
    DIRICHLET_EPS = 1

    bucket_stats = []
    for c in range(n_classes):
        class_ind = jnp.where(y == c, 1, 0)
        bucket_stats.append(
            jax.vmap(lambda b: jnp.zeros((n_buckets,)).at[b].add(class_ind), in_axes=1, out_axes=0)(bucket_indices)
        )
    bucket_stats = jnp.stack(bucket_stats, axis=0)
    bucket_stats = bucket_stats + DIRICHLET_EPS  # Dirichlet prior (to avoid zero probabilities)
    bucket_stats = bucket_stats / bucket_stats.sum(axis=2)[:, :, jnp.newaxis]
    # bucket_stats shape: (n_classes, n_ferns, n_buckets)
    return bucket_stats


def predict_proba(pred_bucket_indices: jnp.ndarray, bucket_stats: jnp.ndarray, prior: jnp.ndarray) -> jnp.ndarray:
    """Predict probabilities with ferns.

    Args:
        pred_bucket_indices: Test points bucket indices of shape (n_samples, n_ferns).
        bucket_stats: Bucket stats of shape (n_classes, n_ferns, n_buckets).
        prior: Prior distribution on classes of shape (n_classes).

    Returns:
        Class probabilities of shape (n_samples, n_classes).

    """
    n_samples = pred_bucket_indices.shape[0]
    n_ferns = pred_bucket_indices.shape[1]
    n_classes = bucket_stats.shape[0]
    # renormalize bucket stats along classes
    # bucket_renormalized = jnp.log(bucket_stats)
    # bucket_renormalized = bucket_renormalized - bucket_renormalized.sum(axis=0)[jnp.newaxis]
    bucket_renormalized = bucket_stats / bucket_stats.sum(axis=0)[jnp.newaxis]
    bucket_renormalized = jnp.log(bucket_renormalized)
    # now bucket_renormalized is a conditional probability P(C | Bucket) for each fern (independently).

    # extract fern probabilities for each sample using the given bucket indices and renormalized buckets
    fern_log_proba = jnp.take_along_axis(bucket_renormalized, pred_bucket_indices.T[np.newaxis], axis=2)
    # fern_proba shape: (n_classes, n_ferns, n_samples)
    fern_log_proba = jnp.transpose(fern_log_proba, (2, 0, 1))
    # fern_proba shape: (n_samples, n_classes, n_ferns)
    assert fern_log_proba.shape[0] == n_samples
    assert fern_log_proba.shape[1] == n_classes
    assert fern_log_proba.shape[2] == n_ferns

    proba = fern_log_proba.sum(axis=2)
    proba = jnp.exp(proba)
    # proba shape: (n_samples, n_classes)
    # multiply by prior and normalize
    proba = proba * prior[jnp.newaxis]
    proba = proba / proba.sum(axis=1)[:, jnp.newaxis]
    return proba


class RandomFernsBlock(BaseBlock):
    """Random Ferns Classifier Block.

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

    def __init__(self, n_groups: int = 10,
                 n_ferns_in_group: int = 20,
                 fern_size: int = 7,
                 kind: str = 'unary',
                 bootstrap: bool = False,
                 n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None):
        """Initialize Random Ferns Block.

        Args:
            n_groups: Number of ferns groups (like a number of estimators).
            n_ferns_in_group: Number of ferns in a group.
            fern_size: Number of tests in fern.
            kind: Kind of tests ('unary' or 'binary').
            bootstrap: Apply data bootstrap or not.
            n_jobs: Number of threads.
            random_state: Random state.

        """
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

    def _classifier_init(self, y):
        check_classification_targets(y)

        self.classes_, y_encoded = jnp.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        y = y_encoded
        return y

    def _parallel_calc_bucket_stats(self, bucket_indices, y, group_data_indices):
        bucket_stats = Parallel(n_jobs=self.n_jobs, prefer='threads')(
            delayed(calculate_bucket_stats)(
                bucket_indices[
                    data_ids,
                    slice(i * self.n_ferns_in_group, (i + 1) * self.n_ferns_in_group)
                ],
                self.n_buckets,
                y[data_ids],
                n_classes=self.n_classes_
            )
            for i, data_ids in enumerate(group_data_indices)
        )
        bucket_stats = jnp.concatenate(bucket_stats, axis=1)
        return bucket_stats

    def _make_ferns(self, X, prng_key):
        if self.kind == 'unary':
            return make_unary_ferns(X, n_ferns=self.n_ferns, fern_size=self.fern_size, key=prng_key)
        elif self.kind == 'binary':
            return make_binary_ferns(X, n_ferns=self.n_ferns, fern_size=self.fern_size, key=prng_key)
        else:
            raise ValueError(f'Wrong kind: {self.kind!r}')

    def _apply_ferns(self, X, ferns):
        if self.kind == 'unary':
            return apply_unary_ferns(X, *ferns)
        elif self.kind == 'binary':
            return apply_binary_ferns(X, *ferns)
        else:
            raise ValueError(f'Wrong kind: {self.kind!r}')

    def fit(self, inputs: BlockInputData) -> 'RandomFernsBlock':
        """Fit the Random Ferns Block.
        The implementation is device-agnostic.

        """
        assert type(inputs['X']) == type(inputs['y'])
        X = inputs['X'].data
        y = inputs['y'].data
        y = self._classifier_init(y)
        np_rng = get_random_generator(self.random_state)
        prng_key = random.PRNGKey(get_rand_int(np_rng))
        prng_key, ferns_key = random.split(prng_key)
        n_samples = X.shape[0]

        ferns = self._make_ferns(X, ferns_key)
        bucket_indices = self._apply_ferns(X, ferns)

        if not self.bootstrap:
            group_data_indices = [slice(None, None) for _ in range(self.n_groups)]
        else:
            keys = random.split(prng_key, self.n_groups)
            group_data_indices = [
                random.choice(keys[i], n_samples, shape=(n_samples,), replace=True)
                for i in range(self.n_groups)
            ]

        if self.n_jobs is None and not self.bootstrap:
            bucket_stats = calculate_bucket_stats(bucket_indices, self.n_buckets, y, n_classes=self.n_classes_)
        else:
            bucket_stats = self._parallel_calc_bucket_stats(bucket_indices, y, group_data_indices)

        _classes, counts = jnp.unique(y, return_counts=True)
        self.prior_ = counts / counts.sum()
        self.ferns_ = ferns
        self.bucket_stats_ = bucket_stats
        return self

    def _predict_proba(self, X):
        group_preds = []
        for group_indices in jnp.split(jnp.arange(self.n_ferns), self.n_groups):
            prediction_bucket_indices = self._apply_ferns(X, [f[group_indices] for f in self.ferns_])
            group_preds.append(
                predict_proba(
                    prediction_bucket_indices,
                    self.bucket_stats_[:, group_indices],
                    prior=self.prior_
                )
            )
        preds = jnp.stack(group_preds, axis=0).mean(axis=0)
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
