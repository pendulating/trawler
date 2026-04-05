# How to add a new CI evaluation benchmark

Adding a new CI-aligned eval means scaffolding a new dagspace that: (1) loads the benchmark data, (2) prompts the model under test, (3) parses outputs, (4) computes metrics, (5) logs to W&B with the standard auto-tags.

## Template: copy `goldcoin_hipaa`

`goldcoin_hipaa` is the cleanest eval template (two-branch: applicability + compliance, parquet IO, LLM judge optional).

```bash
cp -r dagspaces/goldcoin_hipaa dagspaces/my_bench
# then rename package refs inside
```

## Minimum stages

| Stage | Purpose |
|---|---|
| `prepare` | Load raw benchmark → normalize to a unified parquet schema |
| `infer` | Run vLLM over prompts (one row per case) |
| `parse` | Extract structured answer from completion (guided decoding preferred) |
| `score` | Compute metrics vs ground truth |

## Prompt config

Under `dagspaces/my_bench/conf/prompt/<task>.yaml`. Keep CI structure explicit in the prompt (senders, recipients, attributes, transmission principles) — this is what lets the trained models exploit their CI reasoning.

## vLLM call

Always via the shared helper:

```python
from dagspaces.common.vllm_inference import run_vllm_inference

completions = run_vllm_inference(
    model_cfg=cfg.model,
    prompts=prompt_list,
    sampling_params=...,
    guided_decoding_schema=my_schema,  # recommended
)
```

## Metrics + W&B

Use `dagspaces.common.wandb_logger.WandbLogger` so your eval inherits the auto-tags (`bench:my_bench`, `family:<model>`, `finetuned`/`base`, `task:<task>`). Log:
- Aggregate metrics (accuracy, macro-F1, per-class P/R)
- A per-case sample table (`wandb.Table`) — enables qualitative review
- Confusion matrix where applicable

## Hook into `eval_all`

If you want the benchmark to participate in cross-benchmark sweeps, register the dagspace in `dagspaces/eval_all/` (see its pipeline yaml for the pattern).

## Register in paper artifacts

- Add an entry to the benchmark table in `papers/colm26_normative-simulacra/03_methods.tex`
- Add run commands to `EXPERIMENTS.md` and the ablation matrix
- Add W&B project name to the inventory

## Guided decoding note

Per the paper's Reproducibility Statement: the project prefers constrained/guided decoding over regex/string parsers (vs. `fan_goldcoin_2024`). Match this for comparability.
