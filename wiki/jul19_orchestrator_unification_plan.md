# Plan — Unify the eval orchestrators (Finding 1)

**Date:** 2026-07-19 · **Addresses:** `jul19_refactoring.md` Finding 1 (and, as a
by-product, Finding 2 and Finding 10) · **Status:** ✅ **Phase 1 COMPLETE
(2026-07-19)** — all nine eval dagspaces migrated · **Phase 2 DECLINED
(2026-07-19)** — `historical_norms` + `grpo_training` deliberately left as-is

> **Completion note.** Nine eval dagspaces (the seven named below **plus**
> `vlm_geoprivacy_aug` and `ci_heuristic`, which the survey missed) now run on
> the shared loop. Commits `7d79921` (generic loop + tests), `6c34574` (mmlu),
> `5aedadb` (remaining 8). Full suite green (807 passed); parity test
> `test_migrated_dagspaces_expose_hooks` covers all nine. The one remaining
> acceptance item is the **runtime byte-parity baseline** (§8) — a debug run of
> each dagspace on cluster GPU+data to confirm outputs/W&B keys/SLURM job names
> are byte-identical; that needs a cluster job and is left as a manual check.

Replace the seven near-identical eval-dagspace `orchestrator.py` files with one
shared, tested run loop in `dagspaces/common/`, leaving each dagspace a small
stub that supplies only what genuinely differs. This is the highest-leverage
refactor in the repo: ~2400 duplicated lines → one module + seven ~40-line
stubs, and it makes the eval harnesses *provably* identical.

---

## 1. Scope

**In scope (Phase 1) — the eval dagspaces**, whose `orchestrator.py`
files are documented copies of one another and differ only in six parameters.
Nine in total (the seven below **plus** `vlm_geoprivacy_aug` and `ci_heuristic`,
which the original survey missed):

`goldcoin_hipaa`, `vlm_geoprivacy_bench`, `vlm_geoprivacy_aug`, `privacylens`,
`confaide`, `cirl_vignettes`, `mmlu`, `simpleqa_verified`, `ci_heuristic`.

**Declined (Phase 2): `historical_norms` and `grpo_training`.** Decision
2026-07-19 (Matt): these two are *different in purpose* — norm extraction from
fiction vs. policy training — and consolidating them does not make sense.
Technical investigation confirmed it: they are **not** copies of the eval loop
but a distinct, richer "training loop" shape (WANDB_GROUP propagation,
per-stage GPU sanitization, `set_summary` lifecycle, bespoke per-stage table
logging, orchestrator-level metrics). Their NFS result-wait algorithms differ
from the eval `await_slurm_result` **and from each other**, so unifying them
would change waiting behavior on the paper's core training pipeline — which the
`grpo_redesign` wiki requires to stay byte-frozen (keeper path) — for a 2-file
payoff. Wrong abstraction + wrong risk/reward. Left as-is by design.

---

## 2. Invariants (must not change — verified against the code)

The refactor must be **behavior-preserving**. Concretely, for every dagspace:

- **Output paths** unchanged: `output_root = <hydra_output_dir>/<output_subdir>`
  (note `privacylens` uses subdir `privacylens_eval`, *not* `privacylens`).
- **W&B keys, run ids, and grouping** unchanged: `pipeline_run_id(cfg, dagspace=<wandb_dagspace>)`,
  `wandb_run_id = node.wandb_suffix or node.key`, monitor run `stage="orchestrator", run_id="monitor"`.
  privacylens's culture-qualified dagspace (`_perturb_qualified_dagspace`) preserved.
- **SLURM job names** unchanged (`<job_prefix>-<node.key>`): `GoldCoin`, `VLM`,
  `PLens`, `CONFAIDE`, `CIRLVignettes`, `MMLU`, `SimpleQAVerified`.
- **`build_run_config(..., dagspace_name=<dagspace_name>)`** unchanged.
- **`pipeline_manifest.json`** contents and location unchanged.
- **CLI contract** unchanged: `python -m dagspaces.<name>.cli ...` still calls a
  module-level `run_experiment(cfg)`. (`eval_all` invokes children as
  subprocesses via the CLI — it does **not** import `run_experiment` — so the
  internal restructure cannot break it as long as the CLI behaves the same.)
- **`use_srun=False`** for all seven eval dagspaces (they all pass it explicitly).

Acceptance is "the existing test suite stays green **and** a debug run of each
dagspace produces byte-identical outputs/W&B keys/SLURM job names to a
pre-refactor baseline" (§8).

---

## 3. The variation inventory (the only six things that differ)

Verified by diffing all seven files. Everything else — `run_experiment`,
`_serialize_context_data`, `execute_stage_job`, `_get_wandb_logger`,
`_resolve_hydra_output_dir`, the ~120-line SLURM/NFS result-wait block — is
byte-identical modulo the values below.

| Param | goldcoin | vlm | privacylens | confaide | cirl | mmlu | simpleqa |
|---|---|---|---|---|---|---|---|
| `dagspace_name` (build_run_config) | `goldcoin_hipaa` | `vlm_geoprivacy_bench` | `privacylens` | `confaide` | `cirl_vignettes` | `mmlu` | `simpleqa_verified` |
| `output_subdir` | `goldcoin_hipaa` | `vlm_geoprivacy_bench` | **`privacylens_eval`** | `confaide` | `cirl_vignettes` | `mmlu` | `simpleqa_verified` |
| `job_prefix` | `GoldCoin` | `VLM` | `PLens` | `CONFAIDE` | `CIRLVignettes` | `MMLU` | `SimpleQAVerified` |
| `wandb_dagspace` (pipeline_run_id) | `goldcoin` | `vlm_geoprivacy` | `privacylens` *or* `privacylens:<culture>` | `confaide` | `cirl_vignettes` | `mmlu` | `simpleqa_verified` |
| `log_eval_metrics` | confusion-matrix fmt | (its own) | leak-rate fmt | Pearson-r / reject / leak / error fmt | (its own) | (its own) | (its own) |
| `use_srun` | False | False | False | False | False | False | False |

`log_eval_metrics(logger, metrics, stage)` is the **only** substantial
per-benchmark code (10–40 lines each). privacylens's `wandb_dagspace` is a
function of `cfg` (the perturb-culture qualifier); all others are literals.

---

## 4. Target design

### 4.1 `OrchestratorHooks` — the per-dagspace parameter block

New, in `dagspaces/common/orchestrator.py` (alongside the existing shared
helpers; no new module needed):

```python
@dataclass(frozen=True)
class OrchestratorHooks:
    dagspace_module: str        # "dagspaces.goldcoin_hipaa" — for SLURM-worker re-import
    dagspace_name: str          # build_run_config(dagspace_name=...)
    output_subdir: str          # subdir under the hydra output dir
    job_prefix: str             # SLURM job name prefix
    log_eval_metrics: Callable[[Any, Dict[str, Any], str], None]
    wandb_dagspace: Callable[[DictConfig], str] = lambda cfg: ""  # "" => no pipeline_run_id
    use_srun: bool = False
```

`wandb_dagspace` is a callable so privacylens can pass
`_perturb_qualified_dagspace` and the others pass `lambda cfg: "goldcoin"`.

### 4.2 Generic `await_slurm_result` (this is Finding 2)

Extract the ~120-line result-wait block, unchanged, into:

```python
def await_slurm_result(job, cfg, node_key: str) -> StageResult:
    """job.result() with NFS result-pickle recovery + squeue polling fallback."""
```

Pure function of `(job, cfg, node_key)`. This is the riskiest code and the part
that has been hand-patched many times; extracting it once is the point.

### 4.3 Generic `execute_stage_job` and `run_experiment`

```python
def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    hooks = _load_hooks(context_data["dagspace_module"])   # re-import in worker
    registry = _load_registry(context_data["dagspace_module"])
    ...  # identical body to today's, but dagspace_name/wandb come from hooks

def run_experiment(cfg: DictConfig, hooks: OrchestratorHooks) -> None:
    ...  # identical loop to today's, parameterized by hooks;
         # calls await_slurm_result in the launcher branch
```

`_get_wandb_logger` also becomes generic, parameterized by
`hooks.wandb_dagspace(cfg)`:

```python
def make_wandb_logger(cfg, hooks, *, stage, run_id=None, run_config=None):
    wb_config = WandbConfig.from_hydra_config(cfg)
    pipeline_id = pipeline_run_id(cfg, dagspace=hooks.wandb_dagspace(cfg)) \
        if hooks.wandb_dagspace(cfg) else None
    if wb_config.enabled:
        return WandbLogger(cfg, stage=stage, run_id=run_id, run_config=run_config,
                           wandb_id=pipeline_id, resume="allow" if pipeline_id else None)
    return _NoOpLogger(cfg, stage=stage, run_id=run_id, run_config=run_config)
```

### 4.4 Each dagspace's `orchestrator.py` becomes a stub (~40 lines)

```python
# dagspaces/goldcoin_hipaa/orchestrator.py
from dagspaces.common.orchestrator import OrchestratorHooks, run_experiment as _run
from .wandb_logger import WandbConfig, WandbLogger, pipeline_run_id  # noqa: F401 (re-export)

def _log_eval_metrics(logger, metrics, stage): ...   # the ONLY real per-bench code

HOOKS = OrchestratorHooks(
    dagspace_module="dagspaces.goldcoin_hipaa",
    dagspace_name="goldcoin_hipaa",
    output_subdir="goldcoin_hipaa",
    job_prefix="GoldCoin",
    log_eval_metrics=_log_eval_metrics,
    wandb_dagspace=lambda cfg: "goldcoin",
)

def run_experiment(cfg): _run(cfg, HOOKS)
```

`get_stage_registry` stays in each dagspace's `runners/__init__.py` (unchanged);
the generic worker discovers it via `hooks.dagspace_module`.

---

## 5. The SLURM-worker constraint (the one real design risk)

`execute_stage_job` is submitted to submitit, which pickles it **by module
reference** and re-imports it in the worker process. Today each dagspace
submits its *own* `execute_stage_job`, which closes over that module's
`_STAGE_REGISTRY` global and `_get_wandb_logger`.

If we submit a single `common.orchestrator.execute_stage_job`, the worker must
recover the right dagspace's registry + hooks. **Solution:** pass only the
`dagspace_module` *string* in `context_data`; the generic worker does
`importlib.import_module(dagspace_module)` and reads module-level `HOOKS` and
`get_stage_registry()`. Module-level functions (the `log_eval_metrics` and
`wandb_dagspace` callables inside `HOOKS`) pickle by reference, so nothing
un-picklable crosses the wire — only the module path string does.

**Requirements this imposes on each dagspace module:**
- a module-level `HOOKS: OrchestratorHooks` (or `get_orchestrator_hooks()`), and
- `get_stage_registry()` importable from the module (re-export from `runners`).

**Validate early** (§7, step 0) that a submitit worker can re-import a dagspace
module and reconstruct `HOOKS` + registry — this is the assumption the whole
refactor rests on.

---

## 6. Migration strategy — test-first, one dagspace at a time

The metric-trust wiki requires run byte-comparability, so we never refactor
"all seven at once and hope." Order:

1. **Capture baselines.** For each of the seven dagspaces, run
   `pipeline=<eval> runtime.debug=true runtime.sample_n=5 hydra/launcher=null`
   pre-refactor; save `pipeline_manifest.json` + the stage output parquet(s) +
   the W&B run config/keys. These are the golden files.
2. **Add the generic code** (`OrchestratorHooks`, `await_slurm_result`,
   `make_wandb_logger`, generic `execute_stage_job`, generic `run_experiment`)
   in `common/orchestrator.py` **without touching any dagspace yet**. Add
   `tests/common/test_run_experiment.py` (§7). Suite must stay green.
3. **Migrate one dagspace** (start with `mmlu` — smallest, simplest
   `_log_eval_metrics`, no perturb qualifier). Replace its `orchestrator.py`
   with the stub. Re-run its baseline; diff outputs/W&B keys/SLURM job names —
   must be byte-identical. Run the full suite.
4. **Repeat** for `simpleqa_verified`, `confaide`, `cirl_vignettes`,
   `goldcoin_hipaa`, `vlm_geoprivacy_bench`, and finally `privacylens` (last,
   because of the perturb-culture `wandb_dagspace` and the `privacylens_eval`
   subdir — the two genuine wrinkles).
5. Each dagspace migration is its own commit so a regression bisects to one
   dagspace.

---

## 7. Test plan (this is Finding 10)

New `tests/common/test_run_experiment.py`, written **before** step 3:

- **`await_slurm_result`** with a fake `job` object:
  - happy path (`job.result()` returns `{"outputs":..., "metadata":...}`);
  - `(outcome, payload)` tuple unpacking, including `outcome == "error"` → raise;
  - the "has not produced any output" path → result-pickle recovery (write the
    pickle to a tmp path, assert it's read after the wait);
  - the `squeue` fallback (monkeypatch `subprocess.run` to return `R` then empty).
- **`run_experiment` local path** with a stub dagspace (a tiny fake module
  exposing `HOOKS` + a registry of no-op runners): asserts nodes run in
  topological order, outputs register, `pipeline_manifest.json` is written.
- **`run_experiment` SLURM path** with a stubbed `_create_submitit_executor`
  returning a fake job: asserts `await_slurm_result` is used and the result
  propagates.
- **hooks wiring**: `make_wandb_logger` returns `_NoOpLogger` when wandb
  disabled; `wandb_dagspace` callable is honored (test the privacylens
  culture-qualification function in isolation).

Per-dagspace **parity check** (can be a script under `scripts/`, not the suite):
diff the post-refactor baseline against the golden files from step 1.

---

## 8. Acceptance criteria

- [ ] `python -m pytest tests/ -q` green throughout (the project's required bar).
- [ ] New `tests/common/test_run_experiment.py` covers §7.
- [ ] For all seven dagspaces, debug-baseline outputs + W&B keys + SLURM job
      names + `pipeline_manifest.json` are byte-identical pre/post refactor.
- [ ] Each migrated `orchestrator.py` is ≤ ~60 lines and contains no copy of the
      run loop or the SLURM-wait block.
- [ ] `grep -c "result_pickle" dagspaces/*/orchestrator.py` → 0 in the seven
      eval dagspaces (the block now lives once in `common`).
- [ ] No change to any `cli.py`, `runners/`, `stages/`, or `conf/`.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Submitit worker can't re-import dagspace module / reconstruct `HOOKS` | Validate in step 0 with a throwaway submitted job before any migration (§5). Fallback: keep a thin per-dagspace `execute_stage_job` that calls a shared `_execute_stage_job_core(context_data, hooks)`. |
| privacylens perturb-culture `wandb_dagspace` or `privacylens_eval` subdir regresses | Migrate privacylens last; add a focused test for `_perturb_qualified_dagspace` and assert `output_subdir` independently of `dagspace_name`. |
| W&B run-id collision change alters resume behavior | `wandb_dagspace` callable reproduces today's exact `pipeline_run_id` argument; parity check compares W&B run configs. |
| Hidden per-dagspace divergence we didn't diff | The byte-identical baseline diff (§6 step 1/3) catches anything the inventory missed. |
| grpo/historical_norms look tempting to fold in | **Declined (2026-07-19).** They are a distinct "training loop" shape, different in purpose, with divergent result-wait algorithms; consolidating risks the frozen training pipeline for a 2-file payoff. Left as-is by design. |

---

## 10. Follow-ups enabled by this

- **Finding 2** is done as part of this (`await_slurm_result`).
- **Finding 10** is done as part of this (the new test file).
- **Finding 3** (`DataFrameStage` runner base) becomes safe to attempt *after*
  this lands, because the runner↔orchestrator contract is now stable and single.
- **Phase 2 — DECLINED (2026-07-19):** `historical_norms` and `grpo_training`
  are deliberately *not* migrated. They differ in purpose (norm extraction vs.
  policy training) and in orchestration shape (WANDB_GROUP propagation,
  per-stage GPU sanitization, bespoke table logging, and NFS result-wait
  algorithms that differ from the eval loop and from each other). Forcing a
  shared loop would be the wrong abstraction and would change waiting behavior
  on the frozen training pipeline. The high-value dedup (nine eval dagspaces)
  is done; these two stay as-is.

---

## 11. Effort estimate

- Generic code + tests (§4, §7): ~1 day.
- Seven dagspace migrations + parity baselines (§6): ~1 day.
- **Total: ~2 days (M)**, well-isolated and reversible per-commit.
