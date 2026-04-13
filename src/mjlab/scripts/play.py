"""Script to play RL agent with RSL-RL."""

import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


def _is_velocity_command_term(term: object) -> bool:
  return all(
    hasattr(term, attr) for attr in ("vel_command_b", "command", "robot", "compute")
  )


def _resolve_velocity_command_term(
  env: ManagerBasedRlEnv,
  command_name: str | None,
) -> tuple[str | None, object | None]:
  active_names = env.command_manager.active_terms
  if command_name is not None:
    if command_name not in active_names:
      raise ValueError(
        f"Command '{command_name}' not found. Active commands: {active_names}"
      )
    term = env.command_manager.get_term(command_name)
    if not _is_velocity_command_term(term):
      raise ValueError(
        f"Command '{command_name}' is not a velocity command term "
        "(missing vel_command_b/robot fields)."
      )
    return command_name, term

  candidates: list[tuple[str, object]] = []
  for name in active_names:
    term = env.command_manager.get_term(name)
    if _is_velocity_command_term(term):
      candidates.append((name, term))

  if not candidates:
    return None, None

  for preferred in ("base_velocity", "twist"):
    for name, term in candidates:
      if name == preferred:
        return name, term

  if len(candidates) == 1:
    return candidates[0]

  names = [name for name, _ in candidates]
  raise ValueError(
    "Found multiple velocity command terms. Please specify one with "
    f"`--command-name`. Candidates: {names}"
  )


def _install_velocity_command_override(
  term: object,
  cmd_vx: float | None,
  cmd_vy: float | None,
  cmd_wz: float | None,
) -> None:
  if cmd_vx is None and cmd_vy is None and cmd_wz is None:
    return

  original_compute = term.compute

  def _compute_with_override(dt: float) -> None:
    original_compute(dt)
    if cmd_vx is not None:
      term.vel_command_b[:, 0] = cmd_vx
    if cmd_vy is not None:
      term.vel_command_b[:, 1] = cmd_vy
    if cmd_wz is not None:
      term.vel_command_b[:, 2] = cmd_wz
    if hasattr(term, "vel_command_w"):
      term.vel_command_w[:] = term.vel_command_b
    if hasattr(term, "is_standing_env"):
      term.is_standing_env[:] = False

  term.compute = _compute_with_override


class _VelocityDebugPolicy:
  def __init__(
    self,
    policy,
    env,
    command_name: str,
    term: object,
    print_interval: int,
    print_env_id: int,
  ):
    self._policy = policy
    self._env = env
    self._command_name = command_name
    self._term = term
    self._print_interval = max(int(print_interval), 1)
    max_env_id = max(self._env.unwrapped.num_envs - 1, 0)
    self._print_env_id = min(max(int(print_env_id), 0), max_env_id)
    self._call_counter = 0

  def __call__(self, obs):
    actions = self._policy(obs)
    self._call_counter += 1
    if self._call_counter % self._print_interval == 0:
      idx = self._print_env_id
      cmd = self._term.command[idx]
      lin_vel_b = self._term.robot.data.root_link_lin_vel_b[idx]
      ang_vel_b = self._term.robot.data.root_link_ang_vel_b[idx]
      err_vx = float(cmd[0] - lin_vel_b[0])
      err_vy = float(cmd[1] - lin_vel_b[1])
      err_wz = float(cmd[2] - ang_vel_b[2])
      print(
        "[VEL] "
        f"step={self._env.unwrapped.common_step_counter} env={idx} cmd={self._command_name} "
        f"target(vx,vy,wz)=({float(cmd[0]):.3f}, {float(cmd[1]):.3f}, {float(cmd[2]):.3f}) "
        f"actual(vx,vy,wz)=({float(lin_vel_b[0]):.3f}, {float(lin_vel_b[1]):.3f}, {float(ang_vel_b[2]):.3f}) "
        f"error=({err_vx:.3f}, {err_vy:.3f}, {err_wz:.3f})"
      )
    return actions

  def reset(self) -> None:
    self._call_counter = 0
    reset_fn = getattr(self._policy, "reset", None)
    if reset_fn is not None:
      reset_fn()


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  command_name: str | None = None
  """Velocity command term name to override/inspect (e.g. 'base_velocity' or 'twist')."""
  cmd_vx: float | None = None
  """Override commanded linear velocity x (m/s) in play."""
  cmd_vy: float | None = None
  """Override commanded linear velocity y (m/s) in play."""
  cmd_wz: float | None = None
  """Override commanded yaw rate (rad/s) in play."""
  print_actual_vel: bool = False
  """Print target vs actual base velocity during play."""
  print_vel_interval: int = 50
  """Print interval in control steps when --print-actual-vel is enabled."""
  print_env_id: int = 0
  """Which environment index to print when --print-actual-vel is enabled."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  needs_vel_command = (
    cfg.cmd_vx is not None
    or cfg.cmd_vy is not None
    or cfg.cmd_wz is not None
    or cfg.print_actual_vel
  )
  vel_command_name, vel_command_term = _resolve_velocity_command_term(
    env, cfg.command_name
  )
  if needs_vel_command and vel_command_term is None:
    active_names = env.command_manager.active_terms
    raise ValueError(
      "Velocity command override/logging requested, but no compatible velocity "
      f"command term was found. Active commands: {active_names}"
    )

  if vel_command_term is not None and (
    cfg.cmd_vx is not None or cfg.cmd_vy is not None or cfg.cmd_wz is not None
  ):
    _install_velocity_command_override(
      vel_command_term, cfg.cmd_vx, cfg.cmd_vy, cfg.cmd_wz
    )
    print(
      "[INFO]: Override velocity command "
      f"{vel_command_name} to "
      f"vx={cfg.cmd_vx if cfg.cmd_vx is not None else 'unchanged'}, "
      f"vy={cfg.cmd_vy if cfg.cmd_vy is not None else 'unchanged'}, "
      f"wz={cfg.cmd_wz if cfg.cmd_wz is not None else 'unchanged'}"
    )

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  def _decorate_policy(raw_policy):
    if not cfg.print_actual_vel or vel_command_term is None or vel_command_name is None:
      return raw_policy
    print(
      "[INFO]: Printing target vs actual base velocity "
      f"(command={vel_command_name}, env={cfg.print_env_id}, interval={max(cfg.print_vel_interval, 1)} steps)"
    )
    return _VelocityDebugPolicy(
      raw_policy,
      env,
      command_name=vel_command_name,
      term=vel_command_term,
      print_interval=cfg.print_vel_interval,
      print_env_id=cfg.print_env_id,
    )

  policy = _decorate_policy(policy)

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _decorate_policy(_ckpt_runner.get_inference_policy(device=device))

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
