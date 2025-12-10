"""YOLO Edge Optimizer Platform"""
__version__ = "1.0.0"

from .patterns import (
    Preprocessor, StandardPreprocessor, FastPreprocessor,
    PostProcessor, StandardNMS, FastNMS,
    InferenceBackend, TorchBackend, ONNXBackend,
    HardwareIO
)
from .optimizer import ModelOptimizer
from .platform import EdgePlatform

__all__ = [
    'Preprocessor', 'StandardPreprocessor', 'FastPreprocessor',
    'PostProcessor', 'StandardNMS', 'FastNMS',
    'InferenceBackend', 'TorchBackend', 'ONNXBackend',
    'HardwareIO', 'ModelOptimizer', 'EdgePlatform'
]