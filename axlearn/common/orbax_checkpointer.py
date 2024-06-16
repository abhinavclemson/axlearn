"""Orbax Checkpointer"""
from typing import Optional
from absl import logging
from axlearn.common.config import (
    REQUIRED,
    Required,
    Configurable,
    config_class,
)
from axlearn.common.utils import NestedTensor
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions
import orbax.checkpoint as ocp

class OrbaxCheckpointer(Configurable):
    """A Orbax checkpointer that supports various StateStorage implementations."""

    @config_class
    class Config(Configurable.Config):
        """Configures Checkpointer."""

        dir: Required[str] = REQUIRED  # The output directory.

        max_to_keep: Optional[int] = None

        save_interval_steps: Optional[int] = None

        enable_checkpointing: Optional[bool] = False

        use_async: Optional[bool] = True

    def __init__(self, *, cfg: Config):
        super().__init__(cfg)
        self._cfg = cfg
        item_names = ("items",)
        self._manager = None
        if cfg.enable_checkpointing:
            options=CheckpointManagerOptions(
                    create=True,
                    enable_async_checkpointing=cfg.use_async,
                )

            if cfg.max_to_keep:
                options.max_to_keep = cfg.max_to_keep
            if cfg.save_interval_steps:
                options.save_interval_steps=cfg.save_interval_steps

            self.manager = CheckpointManager(
                dir,
                item_names=item_names,
                options=options,
            )

    def save(self, *, step: int, state: NestedTensor):
        if self._manager:
            logging.info("Starting Orbax save operation on step: {step}")
            self._manager.save(
                step, args=ocp.args.StandardSave(item=state)
            )





        