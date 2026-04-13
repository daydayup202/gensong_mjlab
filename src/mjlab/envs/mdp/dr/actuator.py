"""Domain randomization functions for actuators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.actuator import BuiltinPositionActuator, IdealPdActuator
from mjlab.actuator.xml_actuator import XmlActuator
from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import _DEFAULT_ASSET_CFG
from ._types import resolve_distribution

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _resolve_local_ctrl_ids(
  asset: Entity, asset_cfg: SceneEntityCfg, device: str
) -> torch.Tensor:
  """Resolve selected local ctrl indices from SceneEntityCfg."""
  num_ctrl = len(asset.actuator_names)
  actuator_ids = asset_cfg.actuator_ids

  if isinstance(actuator_ids, slice):
    start, stop, step = actuator_ids.indices(num_ctrl)
    ctrl_ids = torch.arange(start, stop, step, device=device, dtype=torch.long)
  elif isinstance(actuator_ids, list):
    ctrl_ids = torch.tensor(actuator_ids, device=device, dtype=torch.long)
  else:
    ctrl_ids = torch.tensor([int(actuator_ids)], device=device, dtype=torch.long)

  if ctrl_ids.numel() == 0:
    return ctrl_ids

  if torch.any(ctrl_ids < 0) or torch.any(ctrl_ids >= num_ctrl):
    raise IndexError(
      f"Actuator indices out of range for entity '{asset_cfg.name}': "
      f"valid [0, {num_ctrl - 1}], got {ctrl_ids.tolist()}"
    )

  return ctrl_ids


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize PD stiffness and damping gains.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    kp_range: (min, max) for proportional gain randomization.
    kd_range: (min, max) for derivative gain randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies default gains by sampled values, "abs" sets
      absolute values.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  selected_local_ctrl_ids = _resolve_local_ctrl_ids(asset, asset_cfg, env.device)
  if selected_local_ctrl_ids.numel() == 0:
    return

  for actuator in asset.actuators:
    local_mask = torch.isin(actuator.ctrl_ids, selected_local_ctrl_ids)
    if local_mask.sum().item() == 0:
      continue
    ctrl_ids = actuator.global_ctrl_ids[local_mask]
    local_cols = local_mask.nonzero(as_tuple=False).squeeze(-1)

    dist = resolve_distribution(distribution)
    kp_samples = dist.sample(
      torch.tensor(kp_range[0], device=env.device),
      torch.tensor(kp_range[1], device=env.device),
      (len(env_ids), len(ctrl_ids)),
      env.device,
    )
    kd_samples = dist.sample(
      torch.tensor(kd_range[0], device=env.device),
      torch.tensor(kd_range[1], device=env.device),
      (len(env_ids), len(ctrl_ids)),
      env.device,
    )

    if isinstance(actuator, BuiltinPositionActuator) or (
      isinstance(actuator, XmlActuator) and actuator.command_field == "position"
    ):
      if operation == "scale":
        default_gainprm = env.sim.get_default_field("actuator_gainprm")
        default_biasprm = env.sim.get_default_field("actuator_biasprm")
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
          default_gainprm[ctrl_ids, 0] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
          default_biasprm[ctrl_ids, 1] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
          default_biasprm[ctrl_ids, 2] * kd_samples
        )
      elif operation == "abs":
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = -kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = -kd_samples

    elif isinstance(actuator, IdealPdActuator):
      assert actuator.stiffness is not None
      assert actuator.damping is not None
      if operation == "scale":
        assert actuator.default_stiffness is not None
        assert actuator.default_damping is not None
        actuator.stiffness[env_ids[:, None], local_cols] = (
          actuator.default_stiffness[env_ids[:, None], local_cols] * kp_samples
        )
        actuator.damping[env_ids[:, None], local_cols] = (
          actuator.default_damping[env_ids[:, None], local_cols] * kd_samples
        )
      elif operation == "abs":
        actuator.stiffness[env_ids[:, None], local_cols] = kp_samples
        actuator.damping[env_ids[:, None], local_cols] = kd_samples

    else:
      raise TypeError(
        f"pd_gains only supports BuiltinPositionActuator, "
        f"XmlActuator (position), and IdealPdActuator, "
        f"got {type(actuator).__name__}"
      )


@requires_model_fields("actuator_forcerange")
def effort_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize actuator effort limits.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    effort_limit_range: (min, max) for effort limit randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies default limits, "abs" sets absolute values.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  selected_local_ctrl_ids = _resolve_local_ctrl_ids(asset, asset_cfg, env.device)
  if selected_local_ctrl_ids.numel() == 0:
    return

  for actuator in asset.actuators:
    local_mask = torch.isin(actuator.ctrl_ids, selected_local_ctrl_ids)
    if local_mask.sum().item() == 0:
      continue
    ctrl_ids = actuator.global_ctrl_ids[local_mask]
    local_cols = local_mask.nonzero(as_tuple=False).squeeze(-1)
    num_actuators = len(ctrl_ids)

    dist = resolve_distribution(distribution)
    effort_samples = dist.sample(
      torch.tensor(effort_limit_range[0], device=env.device),
      torch.tensor(effort_limit_range[1], device=env.device),
      (len(env_ids), num_actuators),
      env.device,
    )

    if isinstance(actuator, BuiltinPositionActuator) or (
      isinstance(actuator, XmlActuator) and actuator.command_field == "position"
    ):
      if operation == "scale":
        default_forcerange = env.sim.get_default_field("actuator_forcerange")
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
          default_forcerange[ctrl_ids, 0] * effort_samples
        )
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          default_forcerange[ctrl_ids, 1] * effort_samples
        )
      elif operation == "abs":
        env.sim.model.actuator_forcerange[
          env_ids[:, None], ctrl_ids, 0
        ] = -effort_samples
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          effort_samples
        )

    elif isinstance(actuator, IdealPdActuator):
      assert actuator.force_limit is not None
      if operation == "scale":
        assert actuator.default_force_limit is not None
        actuator.force_limit[env_ids[:, None], local_cols] = (
          actuator.default_force_limit[env_ids[:, None], local_cols] * effort_samples
        )
      elif operation == "abs":
        actuator.force_limit[env_ids[:, None], local_cols] = effort_samples

    else:
      raise TypeError(
        f"effort_limits only supports BuiltinPositionActuator, "
        f"XmlActuator (position), and IdealPdActuator, "
        f"got {type(actuator).__name__}"
      )
