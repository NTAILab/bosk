from abc import ABC, abstractmethod

class PainterMixin(ABC):
    @abstractmethod
    def draw(self, output_filename: str) -> None:
        ...