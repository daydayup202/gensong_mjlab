from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class WheelModeCommand(CommandTerm):
  """Command term for switching between four-wheel and two-wheel modes."""

  cfg: WheelModeCommandCfg

  def __init__(self, cfg: WheelModeCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._mode = torch.full(
      (self.num_envs, 1), float(cfg.initial_mode), device=self.device, dtype=torch.float32
    )

  @property
  def command(self) -> torch.Tensor:
    return self._mode

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # On reset, the command manager sets command_counter to zero before resampling.
    # This guarantees every episode starts in four-wheel mode.
    is_reset = self.command_counter[env_ids] == 0
    if is_reset.any():
      self._mode[env_ids[is_reset], 0] = float(self.cfg.initial_mode)

    if (~is_reset).any():
      sample_ids = env_ids[~is_reset]
      samples = (
        torch.rand(len(sample_ids), device=self.device) < self.cfg.two_wheel_probability
      )
      self._mode[sample_ids, 0] = samples.float()

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class WheelModeCommandCfg(CommandTermCfg):
  """Configuration for wheel-mode switching command."""

  two_wheel_probability: float = 0.5
  initial_mode: float = 0.0

  def build(self, env: ManagerBasedRlEnv) -> WheelModeCommand:
    return WheelModeCommand(self, env)

  def __post_init__(self):
    if not 0.0 <= self.two_wheel_probability <= 1.0:
      raise ValueError("two_wheel_probability must be in [0.0, 1.0].")
    if self.initial_mode not in (0.0, 1.0):
      raise ValueError("initial_mode must be either 0.0 (4W) or 1.0 (2W).")
