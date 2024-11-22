# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbininan Pöppel

from dataclasses import dataclass, field
from typing import Optional

from ...components.feedforward import FeedForwardConfig
from ..xlstm_block import XLSTMBlock, XLSTMBlockConfig
from .layer import SLSTMLayerConfig


@dataclass
class SLSTMBlockConfig:
    slstm: SLSTMLayerConfig = field(default_factory=SLSTMLayerConfig)
    feedforward: Optional[FeedForwardConfig] = field(default_factory=FeedForwardConfig)

    # we initialize these with None to catch the case where they are not set
    _num_blocks: int = None
    _block_idx: int = None

    def __post_init__(self):
        self.slstm._block_idx = self._block_idx
        self.slstm._num_blocks = self._num_blocks
        self.slstm.__post_init__()
        if self.feedforward is not None:
            self.feedforward.__post_init__()


class SLSTMBlock(XLSTMBlock):
    config_class = SLSTMBlockConfig

    def __init__(self, config: SLSTMBlockConfig):
        super().__init__(
            XLSTMBlockConfig(
                mlstm=None,
                slstm=config.slstm,
                feedforward=config.feedforward,
                _block_idx=config._block_idx,
                _num_blocks=config._num_blocks,
            )
        )
