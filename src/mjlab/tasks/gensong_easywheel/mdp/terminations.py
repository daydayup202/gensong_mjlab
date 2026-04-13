from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_patterns: str | tuple[str, ...],
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  selected = _contact_body_indices(sensor, body_patterns)
  if not selected:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  data = sensor.data
  if data.force_history is not None:
    magnitude = torch.norm(data.force_history[:, selected], dim=-1)
    return (magnitude > force_threshold).any(dim=-1).any(dim=-1)

  force = data.force
  assert force is not None
  magnitude = torch.norm(force[:, selected], dim=-1)
  return (magnitude > force_threshold).any(dim=-1)


def front_wheel_touchdown_in_two_wheel_mode(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_patterns: str | tuple[str, ...],
  force_threshold: float = 1.0,
  command_name: str = "wheel_mode",
  mode_threshold: float = 0.5,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  selected = _contact_body_indices(sensor, body_patterns)
  if not selected:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  command = env.command_manager.get_command(command_name)
  two_wheel_mode = command[:, 0] > mode_threshold

  data = sensor.data
  if data.force_history is not None:
    magnitude = torch.norm(data.force_history[:, selected], dim=-1)
    touched = (magnitude > force_threshold).any(dim=-1).any(dim=-1)
  else:
    force = data.force
    assert force is not None
    magnitude = torch.norm(force[:, selected], dim=-1)
    touched = (magnitude > force_threshold).any(dim=-1)

  return touched & two_wheel_mode
