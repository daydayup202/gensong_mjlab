from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import torch

  from mjlab.envs import ManagerBasedRlEnv


def _expand_range(
  current_range: tuple[float, float],
  target_range: tuple[float, float],
  step_size: float,
) -> tuple[float, float]:
  cur_min, cur_max = float(current_range[0]), float(current_range[1])
  tgt_min, tgt_max = float(target_range[0]), float(target_range[1])
  step = max(float(step_size), 0.0)

  new_min = max(tgt_min, cur_min - step)
  new_max = min(tgt_max, cur_max + step)
  if new_min > new_max:
    mid = 0.5 * (new_min + new_max)
    new_min, new_max = mid, mid
  return (new_min, new_max)


def lin_vel_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "base_velocity",
  rwd_threshold: float = 0.7,
  time_step: float = 1.0e-4,
  max_lin_vel_x: tuple[float, float] = (-1.0, 1.0),
  max_lin_vel_y: tuple[float, float] = (0.0, 0.0),
  reward_term_name: str = "rew_lin_vel_xy",
  success_ratio_threshold: float = 0.8,
) -> dict[str, float]:
  episode_sums = getattr(env.reward_manager, "_episode_sums", None)
  if episode_sums is None or reward_term_name not in episode_sums:
    return {"success_ratio": 0.0}

  rew_per_sec = episode_sums[reward_term_name][env_ids] / env.max_episode_length_s
  if rew_per_sec.numel() == 0:
    return {"success_ratio": 0.0}

  success_ratio = float((rew_per_sec > rwd_threshold).float().mean().item())
  cmd_cfg = env.command_manager.get_term(command_name).cfg

  if success_ratio >= float(success_ratio_threshold):
    cmd_cfg.ranges.lin_vel_x = _expand_range(
      tuple(cmd_cfg.ranges.lin_vel_x), max_lin_vel_x, time_step
    )
    cmd_cfg.ranges.lin_vel_y = _expand_range(
      tuple(cmd_cfg.ranges.lin_vel_y), max_lin_vel_y, time_step
    )

  return {
    "success_ratio": success_ratio,
    "lin_vel_x_min": float(cmd_cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": float(cmd_cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": float(cmd_cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": float(cmd_cfg.ranges.lin_vel_y[1]),
  }


def ang_vel_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "base_velocity",
  rwd_threshold: float = 0.6,
  time_step: float = 1.0e-4,
  max_ang_vel_z: tuple[float, float] = (-0.5, 0.5),
  reward_term_name: str = "rew_ang_vel_z",
  success_ratio_threshold: float = 0.75,
) -> dict[str, float]:
  episode_sums = getattr(env.reward_manager, "_episode_sums", None)
  if episode_sums is None or reward_term_name not in episode_sums:
    return {"success_ratio": 0.0}

  rew_per_sec = episode_sums[reward_term_name][env_ids] / env.max_episode_length_s
  if rew_per_sec.numel() == 0:
    return {"success_ratio": 0.0}

  success_ratio = float((rew_per_sec > rwd_threshold).float().mean().item())
  cmd_cfg = env.command_manager.get_term(command_name).cfg

  if success_ratio >= float(success_ratio_threshold):
    cmd_cfg.ranges.ang_vel_z = _expand_range(
      tuple(cmd_cfg.ranges.ang_vel_z), max_ang_vel_z, time_step
    )

  return {
    "success_ratio": success_ratio,
    "ang_vel_z_min": float(cmd_cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": float(cmd_cfg.ranges.ang_vel_z[1]),
  }
