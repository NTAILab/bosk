"""Visitors for group modifications.
"""
from functools import singledispatchmethod
from typing import Literal
from .base import BaseVisitor
from ..block.slot import BlockGroup
from ..block import BaseBlock


ModificationAction = Literal['add'] | Literal['remove']


class ModifyGroupVisitor(BaseVisitor):
    """Visitor for block groups modification, given a group and action.
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
