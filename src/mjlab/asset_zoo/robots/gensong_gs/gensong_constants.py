"""Local mjlab asset config for the Gensong gs.xml robot."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  BuiltinVelocityActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

GENSONG_GS_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "gensong_gs" / "xmls" / "gs.xml"
)
assert GENSONG_GS_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GENSONG_GS_XML))


GENSONG_UPPER_BODY_JOINTS = (
  "joint_head",
  "joint_waist1",
  "joint_waist2",
  "joint_left_arm_1",
  "joint_left_arm_2",
  "joint_left_arm_3",
  "joint_left_arm_4",
  "joint_left_arm_5",
  "joint_right_arm_1",
  "joint_right_arm_2",
  "joint_right_arm_3",
  "joint_right_arm_4",
  "joint_right_arm_5",
)
GENSONG_LEG_JOINTS = (
  "joint_left_leg_1",
  "joint_left_leg_2",
  "joint_left_leg_3",
  "joint_left_leg_4",
  "joint_left_leg_5",
  "joint_right_leg_1",
  "joint_right_leg_2",
  "joint_right_leg_3",
  "joint_right_leg_4",
  "joint_right_leg_5",
)
GENSONG_WHEEL_1_JOINTS = ("joint_left_wheel_1", "joint_right_wheel_1")
GENSONG_WHEEL_2_JOINTS = ("joint_left_wheel_2", "joint_right_wheel_2")
GENSONG_NON_WHEEL_JOINTS = GENSONG_UPPER_BODY_JOINTS + GENSONG_LEG_JOINTS
GENSONG_POLICY_POS_JOINTS = GENSONG_LEG_JOINTS
GENSONG_POLICY_VEL_JOINTS = GENSONG_LEG_JOINTS + GENSONG_WHEEL_1_JOINTS
GENSONG_WHEEL_1_BODIES = ("left_wheel_1", "right_wheel_1")
GENSONG_UPPER_BODY_LEFT_BODIES = (
  "left_arm_1",
  "left_arm_2",
  "left_arm_3",
  "left_arm_4",
  "left_arm_5",
)
GENSONG_UPPER_BODY_RIGHT_BODIES = (
  "right_arm_1",
  "right_arm_2",
  "right_arm_3",
  "right_arm_4",
  "right_arm_5",
)
GENSONG_UNDESIRED_CONTACT_BODIES = (
  "body",
  "left_leg_1",
  "right_leg_1",
  "left_leg_2",
  "right_leg_2",
  "left_leg_3",
  "right_leg_3",
  "left_leg_4",
  "right_leg_4",
  "left_leg_5",
  "right_leg_5",
)

DEFAULT_Q_4W: dict[str, float] = {
  "joint_head": 0.0,
  "joint_waist1": 0.0,
  "joint_waist2": 0.0,
  "joint_left_arm_1": 0.0,
  "joint_left_arm_2": 0.0,
  "joint_left_arm_3": 0.0,
  "joint_left_arm_4": 0.0,
  "joint_left_arm_5": 0.0,
  "joint_right_arm_1": 0.0,
  "joint_right_arm_2": 0.0,
  "joint_right_arm_3": 0.0,
  "joint_right_arm_4": 0.0,
  "joint_right_arm_5": 0.0,
  "joint_left_leg_1": 0.0,
  "joint_left_leg_2": 0.0,
  "joint_left_leg_3": 0.0,
  "joint_left_leg_4": math.radians(37.5),
  "joint_left_leg_5": -math.radians(75.0),
  "joint_right_leg_1": 0.0,
  "joint_right_leg_2": 0.0,
  "joint_right_leg_3": 0.0,
  "joint_right_leg_4": -math.radians(37.5),
  "joint_right_leg_5": math.radians(75.0),
  "joint_left_wheel_1": 0.0,
  "joint_right_wheel_1": 0.0,
  "joint_left_wheel_2": 0.0,
  "joint_right_wheel_2": 0.0,
}

GENSONG_GS_HEAD_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("joint_head",),
  stiffness=500.0,
  damping=50.0,
  effort_limit=100.0,
  frictionloss=0.0,
)
GENSONG_GS_WAIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("joint_waist.*",),
  stiffness=665.0,
  damping=66.5,
  effort_limit=133.0,
  armature=2.0385298285,
  frictionloss=8.3756844594,
)
GENSONG_GS_ARM_1_2_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("joint_.*_arm_[12]",),
  stiffness=535.0,
  damping=53.5,
  effort_limit=107.0,
  armature=2.26901,
  frictionloss=16.0,
)
GENSONG_GS_ARM_3_4_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("joint_.*_arm_[34]",),
  stiffness=245.0,
  damping=24.5,
  effort_limit=49.0,
  armature=0.6021,
  frictionloss=4.461,
)
GENSONG_GS_ARM_5_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("joint_.*_arm_5",),
  stiffness=67.5,
  damping=6.75,
  effort_limit=13.5,
  armature=0.134112,
  frictionloss=2.002,
)
GENSONG_GS_LEG_1_2_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("joint_.*_leg_[12]",),
  stiffness=665.0,
  damping=66.5,
  effort_limit=133.0,
  armature=2.0385298285,
  frictionloss=8.3756844594,
)
GENSONG_GS_LEG_3_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("joint_.*_leg_3",),
  stiffness=1335.0,
  damping=133.5,
  effort_limit=267.0,
  armature=15.5611502228,
  frictionloss=33.9204772059,
)
GENSONG_GS_LEG_4_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("joint_.*_leg_4",),
  stiffness=1335.0,
  damping=133.5,
  effort_limit=267.0,
  armature=10.9041046071,
  frictionloss=32.0843674861,
)
GENSONG_GS_ANKLE_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("joint_.*_leg_5",),
  stiffness=67.5,
  damping=6.75,
  effort_limit=13.5,
  armature=0.134112,
  frictionloss=2.002,
)
GENSONG_GS_WHEEL_1_ACTUATOR = BuiltinVelocityActuatorCfg(
  target_names_expr=("joint_.*_wheel_1",),
  damping=1.40,
  effort_limit=7.0,
  armature=0.0067368543,
  frictionloss=0.1,
)

GENSONG_GS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GENSONG_GS_HEAD_ACTUATOR,
    GENSONG_GS_WAIST_ACTUATOR,
    GENSONG_GS_ARM_1_2_ACTUATOR,
    GENSONG_GS_ARM_3_4_ACTUATOR,
    GENSONG_GS_ARM_5_ACTUATOR,
    GENSONG_GS_LEG_1_2_ACTUATOR,
    GENSONG_GS_LEG_3_ACTUATOR,
    GENSONG_GS_LEG_4_ACTUATOR,
    GENSONG_GS_ANKLE_ACTUATOR,
    GENSONG_GS_WHEEL_1_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)

GENSONG_GS_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.40),
  joint_pos=DEFAULT_Q_4W,
  joint_vel={".*": 0.0},
)


def get_gensong_gs_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=GENSONG_GS_INIT_STATE.pos,
      rot=GENSONG_GS_INIT_STATE.rot,
      lin_vel=GENSONG_GS_INIT_STATE.lin_vel,
      ang_vel=GENSONG_GS_INIT_STATE.ang_vel,
      joint_pos=DEFAULT_Q_4W.copy(),
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_spec,
    articulation=GENSONG_GS_ARTICULATION,
  )


def _position_scale(effort_limit: float, stiffness: float) -> float:
  return 0.25 * effort_limit / stiffness


GENSONG_GS_POSITION_ACTION_SCALE = {
  "joint_head": _position_scale(100.0, 500.0),
  "joint_waist.*": _position_scale(133.0, 665.0),
  "joint_.*_arm_[12]": _position_scale(107.0, 535.0),
  "joint_.*_arm_[34]": _position_scale(49.0, 245.0),
  "joint_.*_arm_5": _position_scale(13.5, 67.5),
  "joint_.*_leg_[12]": _position_scale(133.0, 665.0),
  "joint_.*_leg_3": _position_scale(267.0, 1335.0),
  "joint_.*_leg_4": _position_scale(267.0, 1335.0),
  "joint_.*_leg_5": _position_scale(13.5, 67.5),
}

GENSONG_GS_EASYWHEEL_JOINT_POS_SCALE = 0.03
GENSONG_GS_EASYWHEEL_JOINT_VEL_SCALE = 8.0
GENSONG_GS_EASYWHEEL_WHEEL_VEL_CLIP = 32.0
