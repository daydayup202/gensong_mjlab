from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _as_patterns(body_patterns: str | tuple[str, ...]) -> tuple[str, ...]:
  if isinstance(body_patterns, str):
    return (body_patterns,)
  return body_patterns


def _contact_body_indices(
  sensor: ContactSensor, body_patterns: str | tuple[str, ...]
) -> list[int]:
  patterns = _as_patterns(body_patterns)
  primary_names: list[str] = []
  for slot in sensor._slots:
    if slot.primary_name not in primary_names:
      primary_names.append(slot.primary_name)
  return [
    idx
    for idx, name in enumerate(primary_names)
    if any(re.fullmatch(pattern, name) for pattern in patterns)
  ]


def _body_positions_in_base_frame(asset: Entity, body_ids) -> torch.Tensor:
  body_pos_w = asset.data.body_link_pos_w[:, body_ids]
  base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(
    -1, body_pos_w.shape[1], -1
  )
  base_pos = asset.data.root_link_pos_w.unsqueeze(1).expand(-1, body_pos_w.shape[1], -1)
  return quat_apply_inverse(base_quat, body_pos_w - base_pos)


def _two_wheel_mode_mask(
  env: ManagerBasedRlEnv,
  command_name: str = "wheel_mode",
  threshold: float = 0.5,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  return (command[:, 0] > threshold).float()


def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
    dim=1,
  )
  return torch.exp(-error / max(std, 1.0e-6) ** 2)


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / max(std, 1.0e-6) ** 2)


def lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_w[:, :2]), dim=1)


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  return torch.sum(
    torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[:, asset_cfg.joint_ids]),
    dim=1,
  )


def feet_distance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  feet_links_name: list[str] | None = None,
  min_feet_distance: float = 0.1,
  max_feet_distance: float = 1.0,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if feet_links_name is not None:
    body_ids, _ = asset.find_bodies(feet_links_name)
  feet_pos = asset.data.body_link_pos_w[:, body_ids]
  distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=1)
  reward = torch.clamp(min_feet_distance - distance, min=0.0, max=1.0)
  reward += torch.clamp(distance - max_feet_distance, min=0.0, max=1.0)
  return reward


def feet_distance_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  feet_links_name: list[str] | None = None,
  min_feet_distance: float = 0.1,
  max_feet_distance: float = 1.0,
  std: float = 0.05,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if feet_links_name is not None:
    body_ids, _ = asset.find_bodies(feet_links_name)
  feet_pos = asset.data.body_link_pos_w[:, body_ids]
  distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=1)
  lower = torch.clamp(min_feet_distance - distance, min=0.0)
  upper = torch.clamp(distance - max_feet_distance, min=0.0)
  denom = max(std, 1.0e-6)
  return torch.square(lower / denom) + torch.square(upper / denom)


def leg_symmetry(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  feet_pos_b = _body_positions_in_base_frame(asset, asset_cfg.body_ids)
  error = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])
  return torch.exp(-torch.square(error) / max(std, 1.0e-6) ** 2)


def upper_body_position_symmetry(
  env: ManagerBasedRlEnv,
  left_body_cfg: SceneEntityCfg,
  right_body_cfg: SceneEntityCfg,
  std: float = 0.3,
  axis_weights: tuple[float, float, float] = (0.5, 1.0, 0.5),
) -> torch.Tensor:
  asset: Entity = env.scene[left_body_cfg.name]
  left_pos_b = _body_positions_in_base_frame(asset, left_body_cfg.body_ids)
  right_pos_b = _body_positions_in_base_frame(asset, right_body_cfg.body_ids)
  pair_count = min(left_pos_b.shape[1], right_pos_b.shape[1])
  left_pos_b = left_pos_b[:, :pair_count]
  right_pos_b = right_pos_b[:, :pair_count]

  err_x = left_pos_b[:, :, 0] - right_pos_b[:, :, 0]
  err_y = left_pos_b[:, :, 1] + right_pos_b[:, :, 1]
  err_z = left_pos_b[:, :, 2] - right_pos_b[:, :, 2]
  wx, wy, wz = axis_weights
  error = wx * torch.square(err_x) + wy * torch.square(err_y) + wz * torch.square(err_z)
  return torch.exp(-torch.mean(error, dim=1) / max(std, 1.0e-6) ** 2)


def same_feet_x_position(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  feet_pos_b = _body_positions_in_base_frame(asset, asset_cfg.body_ids)
  return torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])


def stand_still(
  env: ManagerBasedRlEnv,
  lin_threshold: float = 0.05,
  ang_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  commands = env.command_manager.get_command("base_velocity")
  base_lin_vel = asset.data.root_link_lin_vel_w[:, :2]
  base_ang_vel = asset.data.root_link_ang_vel_w[:, 2]
  lin_mask = (torch.norm(commands[:, :2], dim=1, keepdim=True) < lin_threshold).float()
  ang_mask = (torch.abs(commands[:, 2]) < ang_threshold).float()
  return torch.sum(torch.abs(base_lin_vel) * lin_mask, dim=1) + torch.abs(base_ang_vel) * ang_mask


def base_com_height(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.root_link_pos_w[:, 2] - target_height)


def front_wheel_contact_stability(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_patterns: str | tuple[str, ...],
  force_threshold: float = 1.0,
  command_name: str = "wheel_mode",
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  selected = _contact_body_indices(sensor, body_patterns)
  if not selected:
    return torch.zeros(env.num_envs, device=env.device)

  force = sensor.data.force
  assert force is not None
  in_contact = torch.norm(force[:, selected], dim=-1) > force_threshold
  four_wheel_mask = 1.0 - _two_wheel_mode_mask(env, command_name=command_name)
  return in_contact.float().mean(dim=1) * four_wheel_mask


def front_wheel_lift_height(
  env: ManagerBasedRlEnv,
  body_cfg: SceneEntityCfg,
  min_height: float = 0.08,
  command_name: str = "wheel_mode",
) -> torch.Tensor:
  asset: Entity = env.scene[body_cfg.name]
  front_wheel_height = asset.data.body_link_pos_w[:, body_cfg.body_ids, 2].min(dim=1).values
  two_wheel_mask = _two_wheel_mode_mask(env, command_name=command_name)
  return torch.clamp(front_wheel_height - min_height, min=0.0) * two_wheel_mask


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_patterns: str | tuple[str, ...],
  threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  selected = _contact_body_indices(sensor, body_patterns)
  if not selected:
    return torch.zeros(env.num_envs, device=env.device)
  force = sensor.data.force
  assert force is not None
  contacts = torch.norm(force[:, selected], dim=-1) > threshold
  return contacts.float().sum(dim=1)


class ActionSmoothnessPenalty(ManagerTermBase):
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    super().__init__(env)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    penalty = envs_mdp.action_acc_l2(env)
    penalty = penalty.clone()
    penalty[env.episode_length_buf < 3] = 0.0
    return penalty
