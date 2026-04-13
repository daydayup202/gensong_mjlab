"""Export trained RL policies for deployment/sim2sim."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExportPolicyConfig:
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run (e.g. model_4000.pt)."""
  output_dir: str | None = None
  onnx_filename: str = "policy.onnx"
  torchscript_filename: str = "policy.pt"
  export_onnx: bool = True
  export_torchscript: bool = True
  attach_onnx_metadata: bool = True
  num_envs: int = 1
  device: str | None = None


def _resolve_checkpoint_path(task_id: str, cfg: ExportPolicyConfig) -> Path:
  if cfg.checkpoint_file is not None:
    path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path

  if cfg.wandb_run_path is not None:
    agent_cfg = load_rl_cfg(task_id)
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    checkpoint_path, was_cached = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )
    cache_state = "cached" if was_cached else "downloaded"
    print(f"[INFO]: Resolved checkpoint from W&B ({cache_state}): {checkpoint_path}")
    return checkpoint_path.resolve()

  raise ValueError(
    "Please provide either `--checkpoint-file` or `--wandb-run-path` for export."
  )


def _export_torchscript_from_runner(
  runner: MjlabOnPolicyRunner, output_path: Path
) -> Path:
  model = runner.alg.get_policy().as_onnx(verbose=False)
  model.to("cpu")
  model.eval()

  # Fallback to trace for modules that are not TorchScript-scriptable.
  try:
    scripted = torch.jit.script(model)
  except Exception as e:
    print(f"[WARN] torch.jit.script failed, fallback to trace: {e}")
    dummy_inputs = model.get_dummy_inputs()  # type: ignore[operator]
    if isinstance(dummy_inputs, list):
      dummy_inputs = tuple(dummy_inputs)
    elif not isinstance(dummy_inputs, tuple):
      dummy_inputs = (dummy_inputs,)
    scripted = torch.jit.trace(model, dummy_inputs, strict=False)
    scripted = torch.jit.freeze(scripted.eval())

  output_path.parent.mkdir(parents=True, exist_ok=True)
  scripted.save(str(output_path))
  return output_path


def run_export(task_id: str, cfg: ExportPolicyConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint_path = _resolve_checkpoint_path(task_id, cfg)

  output_dir = (
    Path(cfg.output_dir).expanduser().resolve()
    if cfg.output_dir is not None
    else checkpoint_path.parent.resolve()
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = max(int(cfg.num_envs), 1)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)

  try:
    agent_cfg = asdict(load_rl_cfg(task_id))
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, agent_cfg, log_dir=None, device=device)
    runner.load(
      str(checkpoint_path),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )

    if cfg.export_onnx:
      runner.export_policy_to_onnx(str(output_dir), cfg.onnx_filename)
      onnx_path = output_dir / cfg.onnx_filename
      if cfg.attach_onnx_metadata:
        metadata = get_base_metadata(vec_env.unwrapped, str(checkpoint_path))
        attach_metadata_to_onnx(str(onnx_path), metadata)
      print(f"[INFO]: Exported ONNX policy to {onnx_path}")

    if cfg.export_torchscript:
      ts_path = _export_torchscript_from_runner(
        runner, output_dir / cfg.torchscript_filename
      )
      print(f"[INFO]: Exported TorchScript policy to {ts_path}")
  finally:
    vec_env.close()


def main() -> None:
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    ExportPolicyConfig,
    args=remaining_args,
    default=ExportPolicyConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  if not args.export_onnx and not args.export_torchscript:
    raise ValueError(
      "Nothing to export: set at least one of --export-onnx or --export-torchscript."
    )

  run_export(chosen_task, args)


if __name__ == "__main__":
  main()
