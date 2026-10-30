# pingpong_play: the table-tennis policy RoboJuDo loads by default (sim and real)

- `model_16500.pt`: training checkpoint of run `stage4-fix6-7` (2026-10-26_04-30-16), iteration 16500: the real-robot pick
  (robot randomization; settled gait). Copied from the pingpong folder:
  `playground/humanoid_pingpong/pingpong-version-0/logs/staged_fix6/stage4_robust_gait/2026-10-26_04-30-16_stage4-fix6-7/`
- `params/`: that run's settings. The player reads `params/training.yaml` (planner, swing library, footwork); keep it next to the
  checkpoint. The swing library and the planner tables it names are still read from the pingpong folder (PINGPONG_ROOT).

Default in `robojudo/config/g1/policy/g1_pingpong_policy_cfg.py` (`checkpoint`); another one for a run: `--checkpoint <run>/model_N.pt`.
