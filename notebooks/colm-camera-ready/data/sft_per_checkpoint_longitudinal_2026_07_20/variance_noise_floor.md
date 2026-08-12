## Judge-free benchmark noise floor (2026-07-21 N=3 variance sweep)

| Metric | Rep type | Cells | Median σ | Max σ | Median range | Max range |
|---|---|---|---|---|---|---|
| GoldCoin Comp. | sampled | 44 | 0.58 | 7.08 | 1.16 | 19.32 |
| GoldCoin Appl. | sampled | 44 | 0.55 | 8.2 | 0.96 | 24.2 |
| VLM Q7 | sampled | 24 | 0.17 | 4.73 | 0.32 | 8.94 |
| ConfAIde r 2b | greedy | 44 | 0.74 | 13.69 | 1.37 | 27.24 |
| ConfAIde r 2a | greedy | 44 | 0.52 | 16.7 | 0.96 | 31.61 |
| CIRL-729 Leak↓ | greedy | 28 | 0.47 | 1.58 | 0.88 | 3.07 |
| CIRL-729 Utility | greedy | 28 | 0.45 | 1.02 | 0.85 | 1.87 |
| MMLU Acc | greedy | 44 | 0.05 | 0.27 | 0.08 | 0.55 |
| CIRL-729 Net | greedy | 44 | 0.01 | 0.03 | 0.02 | 0.06 |

*Display units (pct ×100). `sampled` = reps vary `sampling_params.seed` (101/102/103); `greedy` = temp-0 reps, spread is engine nondeterminism. Range = max−min over ≤3 reps. A cross-checkpoint gap below a metric's max range is indistinguishable from re-run noise.*
