__version__ = "0.1.0"
from .api import evaluate_npz
__all__ = ["evaluate_npz", "MolMotionPipeline", "TrajectoryBatch", "__version__"]

def __getattr__(name):
    if name == "MolMotionPipeline":
        from .pipeline import MolMotionPipeline
        return MolMotionPipeline
    if name == "TrajectoryBatch":
        from .trajectory import TrajectoryBatch
        return TrajectoryBatch
    raise AttributeError(name)
