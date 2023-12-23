"""Test Stack and Concat blocks.
"""
from bosk.block.zoo.data_conversion import Stack, Concat
from bosk.data import CPUData, BaseData
import numpy as np


def stack_preserves_inputs_order_test():
    """Check that `Stack` block preserves input order given at init.

    """
    inputs_order = ['first_input', 'a_second_input', 'third_input']
    stack = Stack(inputs_order)

    data = dict(
        first_input=CPUData(np.array([1, 2, 3])),
        a_second_input=CPUData(np.array([4, 5, 6])),
        third_input=CPUData(np.array([7, 8, 9])),
    )

    stack.fit(data)
    result = stack.transform(data)
    output = result['output'].data
    desired_output = np.stack(
        (data[inp].data for inp in inputs_order),
        axis=-1
    )
    print(output)
    assert np.all(output == desired_output)


def concat_preserves_inputs_order_test():
    """Check that `Concat` block preserves input order given at init.

    """
    inputs_order = ['first_input', 'a_second_input', 'third_input']
    concat = Concat(inputs_order)

    data = dict(
        first_input=CPUData(np.array([1, 2, 3])[:, np.newaxis]),
        a_second_input=CPUData(np.array([4, 5, 6])[:, np.newaxis]),
        third_input=CPUData(np.array([7, 8, 9])[:, np.newaxis]),
    )

    concat.fit(data)
    result = concat.transform(data)
    output = result['output'].data
    desired_output = np.concatenate(
        [data[inp].data for inp in inputs_order],
        axis=-1
    )
    print(output)
    assert np.all(output == desired_output)

