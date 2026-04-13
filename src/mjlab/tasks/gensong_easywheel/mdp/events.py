from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import torch

  from mjlab.envs import ManagerBasedRlEnv


def noop(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None, **kwargs) -> None:
  del env, env_ids, kwargs
