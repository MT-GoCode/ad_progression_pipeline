from dataclasses import dataclass, field
from typing import Optional, Callable

@dataclass
class RandomForestPipelineParameters:
    rebalancer: Optional[Callable] = None
    n_estimators: Optional[int] = None         # Number of trees in the forest
    max_depth: Optional[int] = None            # Maximum depth of the trees
    criterion: Optional[str] = None            # Function to measure the quality of a split (e.g., 'gini', 'entropy')