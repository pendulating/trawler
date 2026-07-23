# Benchmarks — reference material

Per-benchmark research references: paper summary, evaluation mechanics, how this project uses it, known gotchas.

| Benchmark | Type | This project's pipeline |
|---|---|---|
| [confaide.md](confaide.md) | Secret-keeping + CI appropriateness (4 tiers) | `dagspaces/confaide/` |
| [privacylens.md](privacylens.md) | Agent-action leakage + QA probing (+ `cirl_protocol` variant) | `dagspaces/privacylens/` |
| [cirl.md](cirl.md) | CIRL-729 synthetic action leakage/utility (judge-free) | `dagspaces/cirl/` |
| [contextreasoner.md](contextreasoner.md) | External PPO baseline model | `dagspaces/common/conf/model/context-reasoner/ppo.yaml` |

See also:
- [dagspaces.md](../dagspaces.md) for the per-pipeline DAGs and judges section
- [overview.md](../overview.md) for the paper's five CI benchmarks (adds GoldCoin-HIPAA, VLM-GeoPrivacy, CI-RL Vignettes)
