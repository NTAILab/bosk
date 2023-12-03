"""Multi-grained scanning blocks.

Multi-grained scanning applies underlying model to a sliding window:

- At the fit stage the data with spatial dimensions is splitted into \
multiple patches (windows) and a new data set is generated by \
repeating a sample label for each window of each sample.
- At the transform stage the model is applied to each window and \
results are concatenated along spatial dimensions.

For example, convolution can be considered as a special case of
multi-grained scanning with linear underlying model corresponding
to the convolution kernel.

There are two different types of multi-grained scanning blocks:

1. N-dimensional Multi-Grained Scanning :py:class:`MultiGrainedScanningNDBlock`, \
as well as pooling :py:class:`PoolingBlock` \
follows the convention that an input data sample is of shape \
`(n_samples, n_channels, spatial_dim_1, ... spatial_dim_k)` \
and output data sample is of shape \
`(n_samples, n_out_channels, out_spatial_dim_1,... out_spatial_dim_k)`.
2. In contrast, :py:class:`MultiGrainedScanning1DBlock` and \
:py:class:`MultiGrainedScanning2DBlock` are applied to standard data \
of shape `(n_samples, n_features)` and return raveled (flattened) result, \
i.e. of shape `(n_samples, n_out_features)`.

The first type supports wider range of parameters, while the second type \
has simpler interface and do not require data reshaping before applying MGS block.

"""

from .multi_grained_scanning_1d import MultiGrainedScanning1D, MultiGrainedScanning1DBlock
from .multi_grained_scanning_2d import MultiGrainedScanning2D, MultiGrainedScanning2DBlock
from .multi_grained_scanning_nd import MultiGrainedScanningND, MultiGrainedScanningNDBlock
from .pooling import Pooling, PoolingBlock


__all__ = [
    'MultiGrainedScanning1D',
    'MultiGrainedScanning2D',
    'MultiGrainedScanningND',
    'Pooling',
    # for backward compatibility:
    'MultiGrainedScanning1DBlock',
    'MultiGrainedScanning2DBlock',
    'MultiGrainedScanningNDBlock',
    'PoolingBlock',
]

