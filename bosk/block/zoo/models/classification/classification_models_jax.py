from __future__ import annotations

from collections import defaultdict

from functools import partial
from typing import Callable, Tuple, Dict, List

import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.tree_util import register_pytree_node_class


from ....base import BaseBlock, TransformOutputData, BlockInputData
from ....meta import BlockMeta, BlockExecutionProperties
from ....slot import InputSlotMeta, OutputSlotMeta
from .....stages import Stages
from .....data import GPUData
from .....utility import get_random_generator, get_rand_int


@partial(vmap, in_axes=(1, None), out_axes=1)
def row_to_nan(X: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Convert a whole row of X to nan with a row mask."""
    return jnp.where(mask > 0, X, jnp.nan)


# @partial(jit, static_argnames="max_splits")
def split_points(
    X: jnp.ndarray, mask: jnp.ndarray, max_splits: int
) -> jnp.ndarray:
    """Generate split points for the data."""
    X = row_to_nan(X, mask)
    delta = 1 / (max_splits + 1)
    quantiles = jnp.nanquantile(
        X, jnp.linspace(delta, 1 - delta, max_splits), axis=0
    )
    return quantiles


def split_mask(
    value: float, col: jnp.ndarray, mask: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    left_mask = jnp.where(col >= value, mask, 0)
    right_mask = jnp.where(col < value, mask, 0)
    return left_mask, right_mask


def compute_score_generic(
    X_col: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    split_value: float,
    score_fn: Callable,
) -> float:
    """Compute the scores of data splits."""
    left_mask, right_mask = split_mask(split_value, X_col, mask)

    left_score = score_fn(y, left_mask)
    right_score = score_fn(y, right_mask)

    n_left = jnp.sum(left_mask)
    n_right = jnp.sum(right_mask)

    avg_score = (n_left * left_score + n_right * right_score) / (
        n_left + n_right
    )

    return avg_score


def make_scoring_function(score_fn: Callable) -> Callable:
    compute_score_specialized = partial(
        compute_score_generic, score_fn=score_fn
    )
    compute_column_scores = vmap(
        compute_score_specialized, in_axes=(None, None, None, 0)
    )

    compute_all_scores = vmap(
        compute_column_scores,
        in_axes=(1, None, None, 1),
        out_axes=1,
    )
    return compute_all_scores


def split_node_generic(
    X: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    max_splits: int,
    compute_all_scores: Callable,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, int]:
    """The algorithm does the following:
    1. Generate split points candidates (N_SPLITS, N_COLS) matrix
    2. For each split point, compute the split score -> (N_SPLITS, N_COLS)
    3. Select the point with the lowest score
    4. Generate two new masks for left and right children nodes
    """
    points = split_points(X, mask, max_splits)
    scores = compute_all_scores(X, y, mask, points)

    split_row, split_col = jnp.unravel_index(
        jnp.nanargmin(scores), scores.shape
    )
    split_value = points[split_row, split_col]
    left_mask, right_mask = split_mask(split_value, X[:, split_col], mask)

    return left_mask, right_mask, split_value, split_col


def make_split_node_function(score_fn: Callable) -> Callable:
    compute_all_scores = make_scoring_function(score_fn)
    split_node_specialized = partial(
        split_node_generic, compute_all_scores=compute_all_scores
    )
    return split_node_specialized


@register_pytree_node_class
class TreeNode:
    """Class representing a node in the tree.
    For jitting to be polyvalent, the same node structure should be used to
    represent nodes, leaves and phantom nodes.
    The boolean `is_leaf` indicates if the node is a leaf and should be used to
    make a prediction.
    If the `mask` only contains zeros, then the node is actually a phantom node,
    that is, positionned below a leaf node and is not used.
    """

    def __init__(
        self,
        mask: jnp.ndarray,
        split_value: float = jnp.nan,
        split_col: int = -1,
        is_leaf: bool = True,
        leaf_value: float = jnp.nan,
        score: float = jnp.nan,
    ):
        self.mask = mask
        self.split_value = split_value
        self.split_col = split_col
        self.is_leaf = is_leaf
        self.leaf_value = leaf_value
        self.score = score

    def tree_flatten(self):
        children = (
            self.mask,
            self.split_value,
            self.split_col,
            self.is_leaf,
            self.leaf_value,
            self.score,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> TreeNode:
        return cls(*children)

    def show(self, rank: int) -> str:
        text = f"n={int(jnp.sum(self.mask[rank]))}\n"
        text += f"loss={self.score[rank]:.2f}\n"
        if self.is_leaf[rank]:
            text += f"value {self.leaf_value[rank]}"
        else:
            text += f"feature {self.split_col[rank]} >= {self.split_value[rank]:.2f}"
        return text


@register_pytree_node_class
class DecisionTreeClassifierBlock(BaseBlock):
    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='X',
                stages=Stages(transform=True),
            ),
            InputSlotMeta(
                name='y',
                stages=Stages(transform=False, transform_on_fit=False),
            )
        ],
        outputs=[
            OutputSlotMeta(
                name='probas',
            )
        ],
        execution_props=BlockExecutionProperties(gpu=True),
    )

    def __init__(
        self,
        n_classes: int,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
        nodes: Dict[int, List[TreeNode]] = None,
    ):
        super().__init__()
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_splits = max_splits
        self._score_fn = accuracy
        self.n_classes = n_classes
        self._loss_fn = partial(entropy, n_classes=n_classes)
        self._value_fn = partial(most_frequent, n_classes=n_classes)
        self.nodes = nodes
        self.split_node = make_split_node_function(self._loss_fn)

    def tree_flatten(self):
        children = [self.nodes]
        aux_data = {
            "n_classes": self.n_classes,
            "min_samples": self.min_samples,
            "max_depth": self.max_depth,
            "max_splits": self.max_splits,
            # "nodes": self.nodes,
            # "value_fn": self._value_fn,
            # "loss_fn": self._loss_fn,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (nodes,) = children
        return cls(**aux_data, nodes=nodes)

    # @jit
    def fit(
        self,
        inputs: BlockInputData,
    ):
        """Fit the model to the data.
        Since this function is functionally pure, the fitted model is returned
        as a result.
        """
        # mask: jnp.ndarray = None
        X = inputs["X"].data
        y = inputs["y"].data
        n_samples = X.shape[0]
        mask = jnp.ones((n_samples,))

        masks = jnp.stack([mask], axis=0)
        self.nodes = defaultdict(list)

        def split_node(carry, x):
            depth, mask = x
            score = self._loss_fn(y, mask)
            value = self._value_fn(y, mask)
            (
                left_mask,
                right_mask,
                split_value,
                split_col,
            ) = self.split_node(X, y, mask, self.max_splits)

            is_leaf = jnp.maximum(depth + 1 - self.max_depth, 0) + jnp.maximum(
                self.min_samples + 1 - jnp.sum(mask), 0
            )
            is_leaf = jnp.minimum(is_leaf, 1).astype(jnp.int8)

            # zero-out child masks if current node is a leaf
            left_mask *= 1 - is_leaf
            right_mask *= 1 - is_leaf

            node = TreeNode(
                mask=mask,
                split_value=split_value,
                split_col=split_col,
                is_leaf=is_leaf,
                leaf_value=value,
                score=score,
            )
            children_mask = jnp.stack([left_mask, right_mask], axis=0)
            return carry, (children_mask, node)

        for depth in range(self.max_depth + 1):
            depths = depth * jnp.ones((masks.shape[0],))
            _, (next_masks, nodes) = lax.scan(
                f=split_node,
                init=None,
                xs=(depths, masks),
            )
            self.nodes[depth] = nodes
            masks = jnp.reshape(next_masks, (-1, n_samples))
        return self

    # @jit
    def transform(
        self,
        inputs: BlockInputData,
    ):
        # print(id(self))
        # print("self.nodes = ", self.nodes)
        X = inputs["X"].data
        mask = None
        X = X.astype("float32")
        n_samples = X.shape[0]
        if mask is None:
            mask = jnp.ones((n_samples,))

        if self.nodes is None:
            raise ValueError("The model is not fitted.")

        @vmap
        def split_and_predict(node, mask):
            left_mask, right_mask = split_mask(
                node.split_value,
                X[:, node.split_col],
                mask,
            )
            predictions = jnp.where(
                mask * node.is_leaf, node.leaf_value, jnp.nan
            )
            child_mask = jnp.stack([left_mask, right_mask], axis=0)
            return child_mask, predictions

        predictions = []
        level_masks = jnp.stack([mask], axis=0)
        for depth in range(self.max_depth + 1):
            next_masks, level_predictions = split_and_predict(
                self.nodes[depth], level_masks
            )
            level_masks = jnp.reshape(next_masks, (-1, n_samples))
            predictions.append(jnp.nansum(level_predictions, axis=0))
        x = jnp.nansum(jnp.stack(predictions, axis=0), axis=0)
        import numpy as np
        one_hot = np.zeros((x.shape[0], self.n_classes))
        for i in range(x.shape[0]):
            one_hot[i, int(x[i])] = 1
        one_hot = jnp.asarray(one_hot)
        return {"probas": GPUData(one_hot)}


# @partial(jit, static_argnames=["n_classes"])
def entropy(y: jnp.ndarray, mask: jnp.ndarray, n_classes: int) -> float:
    """Shannon entropy in bits.
    Returns NaN if no samples.
    """
    n_samples = jnp.sum(mask)
    counts = jnp.bincount(y, weights=mask, length=n_classes)
    probs = counts / n_samples
    log_probs = probs * jnp.log2(probs)
    return -jnp.sum(jnp.where(probs <= 0.0, 0.0, log_probs))


# @partial(jit, static_argnames=["n_classes"])
def most_frequent(y: jnp.ndarray, mask: jnp.ndarray, n_classes: int) -> int:
    counts = jnp.bincount(y.astype(jnp.int8), weights=mask, length=n_classes)
    return jnp.nanargmax(counts)


def accuracy(y_hat: jnp.array, y: jnp.array) -> jnp.array:
    return jnp.mean(y_hat == y)

