from mjlab.tasks.registry import register_mjlab_task

from mjlab.tasks.gensong_easywheel.config.env_cfgs import (
  gensong_easywheel_flat_env_cfg,
)
from mjlab.tasks.gensong_easywheel.config.rl_cfg import (
  gensong_easywheel_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Gensong-EasyWheel-Flat",
  env_cfg=gensong_easywheel_flat_env_cfg(),
  play_env_cfg=gensong_easywheel_flat_env_cfg(play=True),
  rl_cfg=gensong_easywheel_ppo_runner_cfg(),
)
