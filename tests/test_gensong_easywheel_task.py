"""Tests for the standalone Gensong EasyWheel task."""

import io
import warnings
from contextlib import redirect_stderr, redirect_stdout

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots import (
  GENSONG_GS_EASYWHEEL_JOINT_POS_SCALE,
  GENSONG_GS_EASYWHEEL_JOINT_VEL_SCALE,
  get_gensong_gs_robot_cfg,
)
from mjlab.asset_zoo.robots.gensong_gs.gensong_constants import (
  GENSONG_LEG_JOINTS,
  GENSONG_WHEEL_1_JOINTS,
  GENSONG_WHEEL_2_JOINTS,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.tasks.gensong_easywheel.mdp import WheelModeCommandCfg
from mjlab.tasks.registry import load_env_cfg

TASK_ID = "Mjlab-Gensong-EasyWheel-Flat"
SWITCH_TASK_ID = "Mjlab-Gensong-EasyWheel-Flat-Switch-2W"


def test_gensong_robot_compiles() -> None:
  robot = Entity(get_gensong_gs_robot_cfg())
  model = robot.compile()
  assert isinstance(model, mujoco.MjModel)
  assert model.nu == 25
  actuator_names = {model.actuator(i).name for i in range(model.nu)}
  assert set(GENSONG_WHEEL_1_JOINTS).issubset(actuator_names)
  assert actuator_names.isdisjoint(GENSONG_WHEEL_2_JOINTS)


def test_gensong_easywheel_config_shapes() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert "joint_pos" in cfg.actions
  assert "joint_vel" in cfg.actions

  joint_pos = cfg.actions["joint_pos"]
  joint_vel = cfg.actions["joint_vel"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  assert isinstance(joint_vel, JointVelocityActionCfg)
  assert joint_pos.scale == GENSONG_GS_EASYWHEEL_JOINT_POS_SCALE
  assert joint_vel.scale == GENSONG_GS_EASYWHEEL_JOINT_VEL_SCALE
  assert tuple(joint_pos.actuator_names) == GENSONG_LEG_JOINTS
  assert tuple(joint_vel.actuator_names) == GENSONG_WHEEL_1_JOINTS


def test_gensong_easywheel_reward_and_event_names() -> None:
  cfg = load_env_cfg(TASK_ID)
  reward_names = {
    "keep_balance",
    "stand_still",
    "rew_lin_vel_xy",
    "rew_ang_vel_z",
    "rew_same_foot_x_position",
    "pen_action_rate",
    "pen_action_smoothness",
    "pen_non_wheel_joint_deviation",
    "pen_joint_vel_wheel_l2",
    "pen_feet_distance",
    "pen_flat_orientation_l2",
    "pen_base_height",
  }
  event_names = {
    "add_base_mass",
    "add_link_mass",
    "radomize_rigid_body_mass_inertia",
    "robot_physics_material",
    "front_wheel_physics_material",
    "rear_wheel_physics_material",
    "robot_joint_stiffness_and_damping",
    "robot_center_of_mass",
    "reset_robot_base",
    "reset_robot_joints",
    "push_robot",
  }
  assert reward_names.issubset(cfg.rewards.keys())
  assert event_names.issubset(cfg.events.keys())


def test_gensong_easywheel_play_overrides() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  assert cfg.episode_length_s >= 1e9
  assert not cfg.observations["actor"].enable_corruption
  assert "push_robot" not in cfg.events


def test_gensong_easywheel_switch_config() -> None:
  cfg = load_env_cfg(SWITCH_TASK_ID)
  assert "joint_pos" in cfg.actions
  assert "joint_vel" in cfg.actions

  joint_pos = cfg.actions["joint_pos"]
  joint_vel = cfg.actions["joint_vel"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  assert isinstance(joint_vel, JointVelocityActionCfg)
  assert tuple(joint_pos.actuator_names) == GENSONG_LEG_JOINTS
  assert tuple(joint_vel.actuator_names) == GENSONG_WHEEL_1_JOINTS

  assert "base_velocity" in cfg.commands
  assert "wheel_mode" in cfg.commands
  wheel_mode = cfg.commands["wheel_mode"]
  assert isinstance(wheel_mode, WheelModeCommandCfg)
  assert wheel_mode.resampling_time_range == (2.0, 5.0)
  assert wheel_mode.two_wheel_probability == 0.5

  assert "wheel_mode" in cfg.observations["commands"].terms
  assert "rew_front_wheel_contact_4w" in cfg.rewards
  assert "rew_front_wheel_lift_2w" in cfg.rewards
  assert "front_wheel_touchdown_in_two_wheel_mode" in cfg.terminations


def test_gensong_easywheel_switch_play_overrides() -> None:
  cfg = load_env_cfg(SWITCH_TASK_ID, play=True)
  assert cfg.episode_length_s >= 1e9
  assert not cfg.observations["actor"].enable_corruption
  assert "push_robot" not in cfg.events


@pytest.mark.slow
@pytest.mark.parametrize("task_id", [TASK_ID, SWITCH_TASK_ID])
def test_gensong_easywheel_env_smoke(task_id: str) -> None:
  device = get_test_device()
  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = 1

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg=cfg, device=device)
      try:
        env.reset()
        action = torch.zeros(
          (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        env.step(action)
      finally:
        env.close()
