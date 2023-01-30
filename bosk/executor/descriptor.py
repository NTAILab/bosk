from typing import TypeVar, Type
from inspect import signature
from dataclasses import dataclass
from ..stages import Stage
from .handlers import DefaultBlockHandler, DefaultSlotHandler, \
    BaseSlotHandler, BaseBlockHandler

HandlingDescriptorT = TypeVar('HandlingDescriptorT', bound='HandlingDescriptor')


@dataclass(frozen=True, init=True)
class HandlingDescriptor():
    """Dataclass making the code laconic. As all (except of special ones) handlers need
    to know the stage they're running in, it is possible to unite stage and handlers
    in this structure. You can create the descriptor using the factory methods.

    Attributes:
        stage: The computational stage performed by the handlers.
        block_handler: The instance of the block handler.
        slot_handler: The instance of the slot handler.
    """

    stage: Stage
    block_handler: BaseBlockHandler
    slot_handler: BaseSlotHandler

    def __post_init__(self):
        assert self.stage == self.block_handler.stage == self.slot_handler.stage,\
            "Incompatible handlers' stages"

    @classmethod
    def from_classes(cls, stage: Stage, block_handler_cls: Type[BaseBlockHandler] = DefaultBlockHandler, block_handler_kw={},
                     slot_handler_cls: Type[BaseSlotHandler] = DefaultSlotHandler, slot_handler_kw={}) -> HandlingDescriptorT:
        """Static method to make the handling descriptor out of the stage info and handlers' types.

        Args:
            stage: The computational stage
            block_handler_cls: Class (or factory method) which can instantiate the block handler.
            block_handler_kw: Additional to the stage block handler's parameters dictionary.
            slot_handler_cls: Class (or factory method) which can instantiate the slot handler.
            slot_handler_kw: Additional to the stage slot handler's parameters dictionary.
        """
        block_handler = block_handler_cls(stage=stage, **block_handler_kw)
        slot_handler = slot_handler_cls(stage=stage, **slot_handler_kw)
        return HandlingDescriptor(stage=stage, block_handler=block_handler, slot_handler=slot_handler)

    @classmethod
    def from_instances(cls, block_handler: BaseBlockHandler, slot_handler: BaseSlotHandler) -> HandlingDescriptorT:
        """Static method to make the handling descriptor out of instantiated block and slot handlers. 
        The handlers must process the same computational stage.

        Args:
            block_handler: The block handler.
            slot_handler: The slot handler.
        """
        assert block_handler.stage == slot_handler.stage, "Handlers must process the same stage"
        return HandlingDescriptor(stage=block_handler.stage,
                                  block_handler=block_handler, slot_handler=slot_handler)
