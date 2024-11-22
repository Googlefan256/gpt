# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass, field

from ..xlstm_block import XLSTMBlock, XLSTMBlockConfig
from .layer import MLSTMLayerConfig


@dataclass
class MLSTMBlockConfig:
    mlstm: MLSTMLayerConfig = field(default_factory=MLSTMLayerConfig)

    # we initialize these with None to catch the case where they are not set
    _num_blocks: int = None
    _block_idx: int = None

    def __post_init__(self):
        self.mlstm._num_blocks = self._num_blocks
        self.mlstm.__post_init__()


class MLSTMBlock(XLSTMBlock):
    config_class = MLSTMBlockConfig

    def __init__(self, config: MLSTMBlockConfig) -> None:
        super().__init__(
            config=XLSTMBlockConfig(
                mlstm=config.mlstm,
                slstm=None,
                feedforward=None,
                _num_blocks=config._num_blocks,
                _block_idx=config._block_idx,
            )
        )
