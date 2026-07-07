# Run and extend the augmented geoprivacy benchmark

`dagspaces/vlm_geoprivacy_aug` measures how *hypothetical capture contexts*
— textual frames stating what device took the photo — shift a VLM's
normative judgments on VLM-GeoPrivacyBench, relative to an un-framed
baseline. It is a fork of `vlm_geoprivacy_bench`; never modify the original
dagspace, it is the frozen reference implementation.

## Design in one paragraph

The `inpaint_hypotheticals` stage cross-products the 783-image dataset with
the configured variants (a `baseline` control is always inserted), each
variant's frame is rendered into its own prompt, and
`compute_hypothetical_metrics` pairs every variant prediction with the
baseline prediction *on the same image*. Original human labels apply to the
baseline only (they were annotated without device context); everything else
is shift metrics. Conceptually the photo-taker remains the CI **sender**
and the capture device is a **modifier** of the sender — the optional
`bridge` sentence makes that mapping explicit in the prompt.

## Running the arms

```bash
# Primary arm: capture devices with photo-taker bridges (default config)
python -m dagspaces.vlm_geoprivacy_aug.cli model=qwen3-vl-8b/instruct

# Ablation arm: same frames, bridges dropped
python -m dagspaces.vlm_geoprivacy_aug.cli model=qwen3-vl-8b/instruct \
  hypotheticals=capture_devices_raw

# Baseline-only control (parity check against vlm_geoprivacy_bench)
python -m dagspaces.vlm_geoprivacy_aug.cli model=qwen3-vl-8b/instruct \
  hypotheticals=none

# Free-form variant (Q7-only shifts, needs the granularity judge)
python -m dagspaces.vlm_geoprivacy_aug.cli pipeline=freeform_eval_hypotheticals \
  model=qwen3-vl-8b/instruct

# Debug smoke test (local, no SLURM, 5 images x 6 variants = 30 rows)
python -m dagspaces.vlm_geoprivacy_aug.cli model=qwen3-vl-8b/instruct \
  runtime.debug=true runtime.sample_n=5 hydra/launcher=null
```

Both pipelines pin `sampling_params.temperature=0.0`: paired shift metrics
compare the same image across variants, so sampling noise would read as
normative shift.

## Interpreting the metrics

`outputs/compute_hypothetical_metrics/metrics.json`:

| Path | Meaning |
|---|---|
| `per_variant.<id>.Q7.abstention_rate` | Q7="A" rate among parseable rows for that variant |
| `per_variant.<id>.<Q>.label_distribution` | A/B/C proportions |
| `shifts.<id>.<Q>.flip_rate` | fraction of paired images whose prediction changed vs. baseline |
| `shifts.<id>.Q7.mean_ordinal_shift` | mean of (variant − baseline) on A(0)/B(1)/C(2); negative = toward abstention |
| `shifts.<id>.Q7.toward_abstention_rate` / `toward_disclosure_rate` | directional flip fractions |
| `shifts.<id>.Q7.delta_abstention_rate` | variant abstention − baseline abstention, among pairs |
| `baseline.original_label_metrics` | control scored against the original human labels |

Pairs where either side is unparseable are **dropped and counted** in
`metric_provenance` (`n_real` vs `n_total`), never zero-defaulted — the
usual [metric-trust](../metric-trust.md) rules apply.

## Adding a variant

Append to `conf/hypotheticals/capture_devices.yaml` (or a new file). Keep
the parallel three-part frame structure so shifts stay attributable to the
manipulated CI parameters, not wording artifacts:

1. device + who operates/wears it
2. capture manner (deliberate shutter vs. continuous/ambient)
3. bystander notice (can people in view tell they are being recorded?)

```yaml
- id: cctv_fixed            # no dots — ids become dotted metric paths
  dimension: capture_device
  frame: >-
    This photo is a frame from a fixed CCTV camera mounted on a building. ...
  bridge: >-
    For the questions below, the photo-taker is the building operator; ...
  ci_params:                # analysis metadata only, never in the prompt
    sender: building operator
    instrument: fixed CCTV camera
    transmission_principle: continuous stationary capture, posted-notice varies
```

Validation (`hypotheticals.load_variants`) rejects duplicate ids, dots in
ids, missing frames/dimensions, frames or bridges on the baseline, and
unknown positions.

## Adding a dimension

Variants are generic over CI dimensions — a new dimension (e.g. `audience`:
"the footage will be posted publicly" vs. "kept in a private archive") is
just a new YAML file in `conf/hypotheticals/` with `dimension:` set
accordingly. Select it with `hypotheticals=<file>`. Crossing dimensions
(device × audience) needs a composition stage that does not exist yet —
write variants for the cross product by hand if needed before then.

## Invariants the tests enforce

`pytest tests/vlm_geoprivacy_aug/` (no GPU needed):

- A `None`/baseline hypothetical leaves the prompt **byte-identical** to
  the un-augmented benchmark, so `hypotheticals=none` runs are directly
  comparable with `vlm_geoprivacy_bench` results.
- Bridges fold into frames only when `hypotheticals.include_bridges=true`.
- Paired metrics drop unparseable pairs with provenance instead of
  defaulting.
