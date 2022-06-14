from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class Stages:
    fit: bool = True
    transform: bool = True
