"""Module, containing the mixin class with method 
for drawing the computational graph, which was proceeded by some executor.
"""

from abc import ABC, abstractmethod

class PainterMixin(ABC):
    """Mixin class containing method for drawing the computational graph,
    which was proceeded by some executor.
    """
    @abstractmethod
    def draw(self, output_filename: str) -> None:
        """Method for the computational graph drawing.
        
        Args:
            output_filename: Path (containing the filename) where the output graphics file will be saved.
        """
        ...