#!/usr/bin/env python3
"""BABILong StateCache ablation with a pre-QKV route gate enabled.

This intentionally stays as a thin wrapper around ``train_babilong_ablation``:
all data loading, training, checkpoint, and eval behavior stays identical, but
the StateCache gets a feature-wise gate before W_Q/W_K/W_V projection unless
the caller explicitly passes ``--route-gate``.
"""

from __future__ import annotations

import sys

from bam.train_babilong_ablation import main


if "--route-gate" not in sys.argv:
    sys.argv.extend(["--route-gate", "vector"])


if __name__ == "__main__":
    main()
