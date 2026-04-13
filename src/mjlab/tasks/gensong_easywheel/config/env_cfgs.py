"""Standalone mjlab config for the Gensong EasyWheel flat task."""

from __future__ import annotations

import math

from mjlab.asset_zoo.robots import (
  get_gensong_gs_robot_cfg,
)
from mjlab.asset_zoo.robots.gensong_gs.gensong_constants import (
  GENSONG_LEG_JOINTS,
  GENSONG_NON_WHEEL_JOINTS,
  GENSONG_POLICY_POS_JOINTS,
  GENSONG_POLICY_VEL_JOINTS,
  GENSONG_UNDESIRED_CONTACT_BODIES,
  GENSONG_UPPER_BODY_LEFT_BODIES,
  GENSONG_UPPER_BODY_RIGHT_BODIES,
  GENSONG_WHEEL_1_BODIES,
  GENSONG_WHEEL_1_JOINTS,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.gensong_easywheel import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

_FRONT_WHEEL_BODIES = ("left_wheel_2", "right_wheel_2")


def _base_env_cfg() -> ManagerBasedRlEnvCfg:
  contact_sensor = ContactSensorCfg(
    name="contact_forces",
    primary=ContactMatch(mode="body", pattern=r".*", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
    track_air_time=True,
  )

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel, scale=0.25),
    "proj_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GENSONG_POLICY_POS_JOINTS)},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GENSONG_POLICY_VEL_JOINTS)},
      scale=0.05,
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "proj_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="sanitize",
    ),
    "commands": ObservationGroupCfg(
      terms={
        "velocity_commands": ObservationTermCfg(
          func=envs_mdp.generated_commands,
          params={"command_name": "base_velocity"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="sanitize",
    ),
    "obsHistory": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=10,
      flatten_history_dim=True,
      nan_policy="sanitize",
    ),
  }

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=GENSONG_LEG_JOINTS,
      scale=0.03,
      use_default_offset=True,
    ),
    "joint_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=GENSONG_WHEEL_1_JOINTS,
      scale=8.0,
      clip={r"joint_.*_wheel_1": (-32.0, 32.0)},
      use_default_offset=True,
    ),
  }

  commands = {
    "base_velocity": mdp.UniformVelocityCommandCfg(
      entity_name="robot",
      heading_command=False,
      heading_control_stiffness=1.0,
      rel_standing_envs=0.03,
      rel_heading_envs=0.0,
      debug_vis=True,
      resampling_time_range=(3.0, 15.0),
      ranges=mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-0.5, 0.5),
        heading=None,
      ),
      viz=mdp.UniformVelocityCommandCfg.VizCfg(z_offset=1.2, scale=0.5),
    )
  }

  events = {
    "add_base_mass": None,
    "add_link_mass": None,
    "radomize_rigid_body_mass_inertia": None,
    "robot_physics_material": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(r"^(?!.*wheel).*_collision$",)),
        "ranges": (0.4, 1.2),
        "operation": "abs",
        "shared_random": True,
      },
    ),
    "front_wheel_physics_material": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_wheel_2_collision",)),
        "ranges": (0.0, 0.0),
        "operation": "abs",
        "shared_random": True,
      },
    ),
    "rear_wheel_physics_material": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_wheel_1_collision",)),
        "ranges": (1.0, 1.0),
        "operation": "abs",
        "shared_random": True,
      },
    ),
    "robot_joint_stiffness_and_damping": EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(r"(?!.*wheel).*",)),
        "kp_range": (0.95, 1.05),
        "kd_range": (0.95, 1.05),
        "operation": "scale",
      },
    ),
    "robot_center_of_mass": None,
    "reset_robot_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
          "x": (0.0, 0.0),
          "y": (0.0, 0.0),
          "z": (0.0, 0.0),
          "roll": (0.0, 0.0),
          "pitch": (-0.5, 0.5),
          "yaw": (0.0, 0.0),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "randomize_actuator_gains": EventTermCfg(
      func=dr.pd_gains,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(r"(?!.*wheel).*",)),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "push_robot": None,
  }

  rewards = {
    "keep_balance": RewardTermCfg(func=envs_mdp.is_alive, weight=1.5),
    "stand_still": RewardTermCfg(func=mdp.stand_still, weight=-1.5),
    "rew_lin_vel_xy": RewardTermCfg(
      func=mdp.track_lin_vel_xy_exp,
      weight=12.0,
      params={"command_name": "base_velocity", "std": 0.25},
    ),
    "rew_ang_vel_z": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp,
      weight=8.0,
      params={"command_name": "base_velocity", "std": 0.25},
    ),
    "rew_leg_symmetry": RewardTermCfg(
      func=mdp.leg_symmetry,
      weight=1.0,
      params={
        "std": math.sqrt(0.5),
        "asset_cfg": SceneEntityCfg("robot", body_names=GENSONG_WHEEL_1_BODIES),
      },
    ),
    "rew_upper_body_symmetry": RewardTermCfg(
      func=mdp.upper_body_position_symmetry,
      weight=0.5,
      params={
        "left_body_cfg": SceneEntityCfg("robot", body_names=GENSONG_UPPER_BODY_LEFT_BODIES),
        "right_body_cfg": SceneEntityCfg("robot", body_names=GENSONG_UPPER_BODY_RIGHT_BODIES),
        "std": math.sqrt(0.08),
        "axis_weights": (0.5, 1.0, 0.5),
      },
    ),
    "rew_same_foot_x_position": RewardTermCfg(
      func=mdp.same_feet_x_position,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=GENSONG_WHEEL_1_BODIES)},
    ),
    "pen_lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.3),
    "pen_ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.3),
    "pen_joint_torque": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-0.00016,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(r"(?!.*wheel).*",))},
    ),
    "pen_joint_accel": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-1.5e-7),
    "pen_action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
    "pen_non_wheel_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GENSONG_NON_WHEEL_JOINTS)},
    ),
    "pen_non_wheel_joint_deviation": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GENSONG_NON_WHEEL_JOINTS)},
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-0.25,
      params={
        "sensor_name": "contact_forces",
        "body_patterns": GENSONG_UNDESIRED_CONTACT_BODIES,
        "threshold": 10.0,
      },
    ),
    "pen_action_smoothness": RewardTermCfg(
      func=mdp.ActionSmoothnessPenalty,
      weight=-0.02,
    ),
    "pen_flat_orientation_l2": RewardTermCfg(
      func=envs_mdp.flat_orientation_l2,
      weight=-5.0,
    ),
    "pen_feet_distance": RewardTermCfg(
      func=mdp.feet_distance,
      weight=-15.0,
      params={
        "feet_links_name": list(GENSONG_WHEEL_1_BODIES),
        "min_feet_distance": 0.36,
        "max_feet_distance": 0.40,
      },
    ),
    "pen_base_height": RewardTermCfg(
      func=mdp.base_com_height,
      weight=-1.0,
      params={"target_height": 1.368},
    ),
    "pen_joint_vel_wheel_l2": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=0.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GENSONG_WHEEL_1_JOINTS)},
    ),
    "pen_vel_non_wheel_l2": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.08,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GENSONG_NON_WHEEL_JOINTS)},
    ),
  }

  curriculum = {}

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    "base_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={
        "sensor_name": "contact_forces",
        "body_patterns": ("body",),
        "force_threshold": 1.0,
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    decimation=4,
    scene=SceneCfg(
      num_envs=4096,
      env_spacing=2.5,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_gensong_gs_robot_cfg()},
      sensors=(contact_sensor,),
      extent=3.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.005, ccd_iterations=50),
      contact_sensor_maxmatch=256,
      njmax=512,
      nconmax=128,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="body",
      lookat=(0.0, 0.0, 1.0),
      distance=3.5,
      elevation=-20.0,
      azimuth=90.0,
    ),
    episode_length_s=20.0,
    rewards=rewards,
    terminations=terminations,
    commands=commands,
    curriculum=curriculum,
    seed=42,
  )


def gensong_easywheel_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _base_env_cfg()

  if play:
    cfg = _apply_play_overrides(cfg)

  return cfg


def gensong_easywheel_flat_switch_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _base_env_cfg()

  cfg.commands["wheel_mode"] = mdp.WheelModeCommandCfg(
    resampling_time_range=(2.0, 5.0),
    debug_vis=False,
    two_wheel_probability=0.5,
    initial_mode=0.0,
  )

  cfg.observations["commands"].terms["wheel_mode"] = ObservationTermCfg(
    func=envs_mdp.generated_commands,
    params={"command_name": "wheel_mode"},
  )

  cfg.rewards["rew_front_wheel_contact_4w"] = RewardTermCfg(
    func=mdp.front_wheel_contact_stability,
    weight=1.0,
    params={
      "sensor_name": "contact_forces",
      "body_patterns": _FRONT_WHEEL_BODIES,
      "force_threshold": 1.0,
      "command_name": "wheel_mode",
    },
  )
  cfg.rewards["rew_front_wheel_lift_2w"] = RewardTermCfg(
    func=mdp.front_wheel_lift_height,
    weight=2.0,
    params={
      "body_cfg": SceneEntityCfg("robot", body_names=_FRONT_WHEEL_BODIES),
      "min_height": 0.08,
      "command_name": "wheel_mode",
    },
  )

  cfg.terminations["front_wheel_touchdown_in_two_wheel_mode"] = TerminationTermCfg(
    func=mdp.front_wheel_touchdown_in_two_wheel_mode,
    params={
      "sensor_name": "contact_forces",
      "body_patterns": _FRONT_WHEEL_BODIES,
      "force_threshold": 1.0,
      "command_name": "wheel_mode",
      "mode_threshold": 0.5,
    },
  )

  if play:
    cfg = _apply_play_overrides(cfg)

  return cfg


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.scene.num_envs = 32
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  cfg.observations["obsHistory"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}
  return cfg
