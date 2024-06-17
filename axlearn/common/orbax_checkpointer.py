"""Orbax Checkpointer"""
from typing import Optional, Union
from absl import logging
from axlearn.common.config import (
    Configurable,
    config_class,
)
from axlearn.common.utils import NestedTensor, NestedTensorSpec
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions
import orbax.checkpoint as ocp

class OrbaxCheckpointer(Configurable):
    """A Orbax checkpointer that supports various StateStorage implementations."""

    @config_class
    class Config(Configurable.Config):
        """Configures Checkpointer."""

        dir: Optional[str] = ""  # The output directory.

        max_to_keep: Optional[int] = None

        save_interval_steps: Optional[int] = None

        enable_checkpointing: Optional[bool] = False

        use_async: Optional[bool] = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._cfg = cfg
        self.enable_checkpointing = cfg.enable_checkpointing
        self.dir = cfg.dir
        cfg.enable_checkpointing = True # Remove the line
        cfg.save_interval_steps = 10 # Remove the line
        self.item_names = ("items",)
        self._manager = None

    def setup(self):
        cfg = self._cfg
        if cfg.enable_checkpointing:
            options=CheckpointManagerOptions(
                    create=True,
                    enable_async_checkpointing=cfg.use_async,
                )

            if cfg.max_to_keep:
                options.max_to_keep = cfg.max_to_keep
            if cfg.save_interval_steps:
                options.save_interval_steps=cfg.save_interval_steps

            self._manager = CheckpointManager(
                self.dir,
                item_names=self.item_names,
                options=options,
            )
        if self._manager:
            return True
        return False


    def save(self, step: int, state: Union[NestedTensor, NestedTensorSpec]):
        if self._manager:
            logging.info("Starting Orbax save operation on step: {step}")
            self._manager.save(
                step, args=ocp.args.StandardSave(item=state)
            )





        