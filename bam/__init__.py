"""BAM (bolt-on attention memory) package.

Public API:
  - StateCache: the bounded-ring-buffer cross-attention module.
  - BAMGenerator: autoregressive generator that wires StateCache into
    Falcon Mamba via forward hooks and maintains MambaCache across steps.
"""

from bam.cache import StateCache
from bam.generator import BAMGenerator

__all__ = ["StateCache", "BAMGenerator"]
