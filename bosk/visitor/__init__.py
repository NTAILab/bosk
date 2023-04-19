# from .base import BaseVisitor
# from .group import ModificationAction, ModifyGroupVisitor

# __all__ = [
#     "BaseVisitor",
#     "ModificationAction",
#     "ModifyGroupVisitor",
# ]

# circular import! BaseBlock->BaseVisitor: visitor.__init__->visitor.group->BaseBlock