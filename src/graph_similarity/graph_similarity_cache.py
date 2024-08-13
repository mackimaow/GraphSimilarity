
from typing import Generic, TypeVar, List
from dataclasses import dataclass

T = TypeVar("T")

@dataclass
class GraphSimilarityCache(Generic[T]):
    number_of_samples: int
    characteristic_averages: List[T]
    def clear(self):
        self.number_of_samples = 0
        self.cache = {}
    
    @classmethod
    def new(cls) -> "GraphSimilarityCache[T]":
        return GraphSimilarityCache(0, [])