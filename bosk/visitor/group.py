"""Visitors for group modifications.
"""
from functools import singledispatchmethod
from typing import Literal
from .base import BaseVisitor
from ..block import BaseBlock
from ..block.base import BlockGroup


ModificationAction = Literal['add', 'remove']


class ModifyGroupVisitor(BaseVisitor):
    """Visitor for block groups modification, given a group and action.

    Args:
        action: Action to perform (add block to group or remove from it).
        group: The group to modify.

    """
    def __init__(self, action: ModificationAction, group: BlockGroup):
        self.action = action
        self.group = group

    @singledispatchmethod
    def visit(self, obj):
        pass  # ignore extra entities

    @visit.register
    def _(self, block: BaseBlock):
        if self.action == 'add':
            self.group.add(block)
        elif self.action == 'remove':
            self.group.remove(block)
        else:
            raise NotImplementedError()
