"""Runtime patches for TRL colocate vLLM generation.

TRL builds its colocate vLLM engine inside ``VLLMGeneration._init_vllm`` and
gives no passthrough for engine keyword arguments. It also pushes each training
parameter into that engine with ``VLLMGeneration._push_param_to_vllm``. Two
model families need different corrections at these two points, and the cluster
needs a third correction at more than one GPU.

This module holds those three patches. Call :func:`apply_colocate_patches` one
time, before you build the ``GRPOTrainer``.

Each patch is a wrapper around a TRL method. The wrappers install in this
order:

1. ``disable_custom_all_reduce`` — model-agnostic, applies at TP>1.
2. ``qwen3.5`` — composite config plus a weight-sync name prefix.
3. ``gemma-4`` — a text-only weight-sync filter.

**The order of 1 and 2 is a hard requirement.** Both patch ``_init_vllm``, and
step 2 captures the result of step 1 as its own original. If you install them in
the opposite order, the all-reduce patch is lost.

Steps 2 and 3 are mutually exclusive, because a model has one family.

A patch that fails prints a warning and returns ``False``. It does not stop
training. This keeps the behavior that this code had inside
``run_grpo_training_stage()``.

The long comments in the private functions below are the original architecture
notes. They are unchanged, because they record verified vLLM and transformers
behavior.
"""

from __future__ import annotations

from typing import Any

__all__ = ["apply_colocate_patches"]


def apply_colocate_patches(
    *,
    model: Any,
    model_family: str,
    model_source: str,
    use_vllm: bool,
    vllm_mode: str,
    grpo_cfg: Any,
) -> list[str]:
    """Install the colocate patches that this run needs.

    Args:
        model: The training model, after TRL and PEFT wrap it. The gemma-4
            patch reads its class name for a log line.
        model_family: The value of ``model.model_family`` from the config.
        model_source: The value of ``model.model_source`` from the config.
            This is the original model zoo path, not the merged directory.
        use_vllm: True if this run generates with vLLM.
        vllm_mode: ``"colocate"`` or ``"server"``.
        grpo_cfg: The ``training.grpo`` config node.

    Returns:
        The names of the patches that installed correctly, in install order.
        A caller can log this list. An empty list is correct for a run that
        needs no patch.
    """
    applied: list[str] = []

    if _install_disable_custom_all_reduce(use_vllm, vllm_mode, grpo_cfg):
        applied.append("disable_custom_all_reduce")

    family = model_family.lower()

    if "qwen3.5" in family and _install_qwen35_composite_sync(model_source):
        applied.append("qwen3.5")

    if "gemma-4" in family and _install_gemma4_text_only_sync(model):
        applied.append("gemma-4")

    return applied


def _install_disable_custom_all_reduce(
    use_vllm: bool, vllm_mode: str, grpo_cfg: Any
) -> bool:
    """Inject ``disable_custom_all_reduce=True`` into the colocate engine.

    This patch is model-agnostic. It does nothing at TP=1, unless the config
    asks for it.
    """
    # ── Multi-GPU colocate: disable vLLM custom all-reduce (PCIe A6000 lane) ──
    # TRL's colocate LLM(...) construction (VLLMGeneration._init_vllm) hardcodes
    # its engine kwargs and exposes NO passthrough — GRPOConfig has no
    # `vllm_disable_custom_all_reduce` field — so the flag can only be injected by
    # wrapping vllm.LLM.__init__ for the duration of _init_vllm. This is MANDATORY
    # at TP>1 on this cluster: P2P is disabled cluster-wide (server.env /
    # slurm_train_2x NCCL_P2P_DISABLE=1), and vLLM's custom all-reduce probe then
    # crashes at engine init with `custom_all_reduce.cuh 'invalid argument'` (the
    # 2026-07-24 probe-calibration precedent). Mirrors the shared inference util
    # (dagspaces/common/vllm_inference.py: `setdefault disable_custom_all_reduce`
    # at TP>1). Default: on when TP>1, overridable via the config knob. Additive,
    # model-agnostic, no-op at TP=1 unless explicitly requested. Installs BEFORE
    # the qwen3.5 block so the two _init_vllm wrappers compose (qwen captures this
    # wrapper as its `orig`); for gemma-4 only this wrapper runs.
    if not (use_vllm and vllm_mode == "colocate"):
        return False

    _tp = int(grpo_cfg.get("vllm_tensor_parallel_size", 1) or 1)
    _disable_car = grpo_cfg.get("vllm_disable_custom_all_reduce")
    if _disable_car is None:
        _disable_car = _tp > 1
    if not _disable_car:
        return False

    try:
        from trl.generation.vllm_generation import VLLMGeneration as _VG_car
        _orig_init_vllm_car = _VG_car._init_vllm

        def _init_vllm_disable_custom_ar(self_vllm,
                                         __orig=_orig_init_vllm_car):
            from vllm import LLM as _LLM_car
            _real_llm_init = _LLM_car.__init__

            def _llm_init_no_custom_ar(llm_self, *args, **kwargs):
                # Only the colocate engine passes distributed_executor_
                # backend="external_launcher"; setdefault so an explicit
                # caller value (or the qwen composite wrapper's kwargs)
                # still wins on other keys.
                kwargs.setdefault("disable_custom_all_reduce", True)
                return _real_llm_init(llm_self, *args, **kwargs)

            _LLM_car.__init__ = _llm_init_no_custom_ar
            try:
                __orig(self_vllm)
            finally:
                _LLM_car.__init__ = _real_llm_init

        _VG_car._init_vllm = _init_vllm_disable_custom_ar
        print("[grpo_training] Colocate TP>1: injected "
              "disable_custom_all_reduce=True into vLLM engine init "
              f"(TP={_tp}, PCIe-A6000 P2P-disabled lane)")
        return True
    except Exception as _e:
        print("[grpo_training] Warning: could not inject "
              f"disable_custom_all_reduce: {_e}")
        return False


def _install_qwen35_composite_sync(model_source: str) -> bool:
    """Give vLLM the composite Qwen3.5 config, and prefix the sync names.

    Args:
        model_source: The original model zoo path. The patch reloads the
            composite config from this path, not from the merged directory.
    """
    # Qwen3.5 is natively multimodal. TRL loads the CausalLM (text-only) for
    # training, but vLLM needs the composite Qwen3_5Config (with vision_config)
    # to initialize the full model from merged_dir.  Monkey-patch to reload
    # the config from the original model zoo path.
    try:
        from trl.generation.vllm_generation import VLLMGeneration
        _orig_init_vllm = VLLMGeneration._init_vllm

        def _patched_init_vllm(self_vllm,
                               _zoo_path=model_source):
            from vllm import LLM as _LLM
            _orig_LLM = _LLM.__init__

            def _llm_init_with_composite_config(llm_self, *args, **kwargs):
                # Ensure vLLM gets the composite Qwen3_5Config (with
                # vision_config) even though merged_dir was saved from
                # CausalLM with Qwen3_5TextConfig.
                def _ensure_composite(config):
                    if hasattr(config, "vision_config") and config.vision_config is not None:
                        return config
                    from transformers import AutoConfig as _AC
                    try:
                        return _AC.from_pretrained(_zoo_path, trust_remote_code=True)
                    except Exception:
                        return config
                kwargs["hf_overrides"] = _ensure_composite
                return _orig_LLM(llm_self, *args, **kwargs)

            _LLM.__init__ = _llm_init_with_composite_config
            try:
                _orig_init_vllm(self_vllm)
            finally:
                _LLM.__init__ = _orig_LLM

        VLLMGeneration._init_vllm = _patched_init_vllm

        # Colocate weight-sync name remap (vLLM 0.25 composite VLM arch).
        # TRL's sync_weights pushes the text training model's params (loaded
        # as Qwen3_5ForCausalLM → names `model.*` / `lm_head.*`) one-by-one
        # into the vLLM engine model, which is the COMPOSITE
        # Qwen3_5ForConditionalGeneration (the composite-config patch above).
        # That composite nests the text stack under `language_model.`
        # (`self.language_model = Qwen3_5ForCausalLM(prefix="language_model")`),
        # so a bare `model.layers.…` name has no target and load_weights
        # raises "no module or parameter named 'model'". Patch
        # _push_param_to_vllm (runs at SYNC time, engine awake — the engine
        # model is asleep/None right after _init_vllm, so patching the model
        # object there is unreliable) to prefix text names into the composite
        # namespace. Deterministic + self-validating: a wrong name still
        # raises (vLLM checks every name), so this cannot silently mis-route.
        # Vision params are never pushed (LoRA is text-only), so untouched.
        _orig_push = VLLMGeneration._push_param_to_vllm

        def _push_param_to_vllm_lm_prefix(self_vllm, name, param):
            if self_vllm.mode == "colocate" and (
                name.startswith("model.") or name.startswith("lm_head.")
            ):
                name = "language_model." + name
            return _orig_push(self_vllm, name, param)

        VLLMGeneration._push_param_to_vllm = _push_param_to_vllm_lm_prefix
        print(f"[grpo_training] Patched TRL vLLM init for Qwen3.5 "
              f"(composite config from {model_source}) + "
              f"colocate weight-sync remap (model.*/lm_head.* → language_model.*)")
        return True
    except Exception as e:
        print(f"[grpo_training] Warning: failed to patch TRL vLLM init: {e}")
        return False


def _install_gemma4_text_only_sync(model: Any) -> bool:
    """Push only the text stack during a gemma-4 colocate weight sync.

    Args:
        model: The training model. The function reads its class name and the
            class name under the PEFT wrappers, for one log line.
    """
    # ── gemma-4: composite-arch weight sync (the analogue to the qwen3.5 block) ──
    # gemma-4 is ALSO a composite VLM (Gemma4UnifiedForConditionalGeneration), but
    # the qwen fix does NOT transfer — it would break the sync. Two arch facts
    # (verified 2026-07-24, cu129 venv) make gemma the mirror image of qwen here:
    #
    #  1. `AutoModelForCausalLM.from_pretrained(merged_dir)` returns the FULL
    #     composite for gemma-4 (Gemma4UnifiedConfig is unmapped in the CausalLM
    #     registry → AutoModel falls back to the config's own
    #     `Gemma4UnifiedForConditionalGeneration`). So the TRAINING model's params
    #     are already composite-namespaced: `model.language_model.layers.…`
    #     (embeddings are tied — there is NO separate `lm_head.*`). Qwen, by
    #     contrast, loads TEXT-ONLY (`model.*` / `lm_head.*`) and needed manual
    #     re-prefixing into `language_model.…`.
    #  2. vLLM's Gemma4UnifiedForConditionalGeneration.load_weights applies an
    #     `hf_to_vllm_mapper` that already maps `model.language_model.` →
    #     `language_model.model.`, `lm_head.` → `language_model.lm_head.`, and a
    #     `model` catch-all → `language_model.model`. Qwen's Qwen3_5 mapper has NO
    #     such prefix rule (only stacked-qkv entries), which is why qwen needed the
    #     manual `_push_param_to_vllm` prefix and gemma does NOT.
    #
    # TRL's colocate `_push_param_to_vllm` calls `model.load_weights([(name,param)])`
    # — the SAME mapper-applying path used for the initial on-disk load (which is
    # known-good: gemma-4-12b loads & generates, canonical-models.md). So the sync
    # names resolve natively; a qwen-style manual prefix would DOUBLE-prefix
    # (`language_model.model.language_model.…`) and every push would raise.
    #
    # Two additive contributions (NO text-name rewriting — the mapper does that):
    #  (a) a TEXT-ONLY sync filter. Because the composite training model also
    #      carries FROZEN vision/audio params that the vLLM encoder-free
    #      Gemma4Unified lacks (e.g. `model.embed_vision.pos_embedding`), pushing
    #      them raises "no module or parameter named …". LoRA is text-only and
    #      vision is already loaded at engine init, so we push only
    #      `model.language_model.*` (tied `lm_head.*`) and skip the rest — the
    #      qwen path got this for free (text-only training model). Live-confirmed
    #      2026-07-24: the TP=2 smoke reached sync and hit exactly this vision
    #      param; the m0 OOM had blocked the sync entirely before then.
    #  (b) a bounded diagnostic that logs the first pushed/skipped names and
    #      annotates any residual TEXT-param failure with the gemma naming context.
    try:
        # Dig past PEFT/LoRA wrappers to the underlying arch class for the log.
        _under = model
        for _attr in ("base_model", "model"):
            _under = getattr(_under, _attr, _under)
        print(f"[grpo_training] gemma-4 composite weight-sync: training model "
              f"= {type(model).__name__} (arch {type(_under).__name__}); "
              f"relying on vLLM's Gemma4Unified hf_to_vllm_mapper "
              f"(model.language_model.* → language_model.model.*) — no manual "
              f"prefix (unlike qwen3.5).")

        # This branch runs ONLY for gemma-4 (never qwen), so the freshly
        # imported method is the unpatched TRL original.
        from trl.generation.vllm_generation import VLLMGeneration as _VG_g
        _orig_push_g = _VG_g._push_param_to_vllm
        _g_diag = {"n": 0, "nskip": 0}

        def _push_param_to_vllm_gemma_diag(self_vllm, name, param,
                                           __orig=_orig_push_g):
            # gemma-4 colocate sync: push ONLY the text stack
            # (`model.language_model.*` / tied `lm_head.*`) and SKIP vision /
            # audio params. The merged TRAINING model is the FULL multimodal
            # Gemma4Unified, so TRL's sync_weights iterates its vision/audio
            # embedder+tower params too — but the vLLM ENCODER-FREE
            # Gemma4Unified variant lacks most of them (verified live 07-24:
            # pushing `model.embed_vision.pos_embedding` raised "no module or
            # parameter named 'embed_vision.pos_embedding' … available:
            # {'embed_vision.embedding_projection.weight'}"). LoRA is
            # text-only and every vision/audio weight is FROZEN — identical to
            # base and already loaded by vLLM at engine init — so they never
            # need re-syncing. Skipping them reproduces the qwen path's
            # behaviour, whose text-only training model simply never yielded
            # vision params. The text names still flow through vLLM's
            # hf_to_vllm_mapper (`model.language_model.`→`language_model.model.`,
            # verified: sync[0..] land) — NO manual prefixing (that is the
            # gemma-vs-qwen distinction).
            if self_vllm.mode == "colocate" and not (
                name.startswith("model.language_model.")
                or name.startswith("lm_head.")
            ):
                if _g_diag["nskip"] < 5:
                    print(f"[grpo_training] gemma-4 sync: SKIP non-text param "
                          f"'{name}' (frozen vision/audio; absent from vLLM "
                          f"encoder-free Gemma4Unified)")
                    _g_diag["nskip"] += 1
                return None
            if self_vllm.mode == "colocate" and _g_diag["n"] < 5:
                print(f"[grpo_training] gemma-4 sync[{_g_diag['n']}]: "
                      f"push '{name}' shape={tuple(param.shape)}")
                _g_diag["n"] += 1
            try:
                return __orig(self_vllm, name, param)
            except Exception as _e:
                raise RuntimeError(
                    f"gemma-4 colocate weight sync failed on TEXT param "
                    f"'{name}'. Expected vLLM's Gemma4Unified hf_to_vllm_mapper "
                    f"to map 'model.language_model.*'→'language_model.model.*'. "
                    f"If this is a name-not-found error, the mapper contract "
                    f"changed — do NOT copy qwen's prefix (it double-prefixes). "
                    f"Original error: {_e!r}"
                ) from _e

        _VG_g._push_param_to_vllm = _push_param_to_vllm_gemma_diag
        print("[grpo_training] Installed gemma-4 colocate weight-sync filter "
              "(text-only push; vision/audio skipped)")
        return True
    except Exception as e:
        print(f"[grpo_training] Warning: failed to install gemma-4 sync "
              f"diagnostic: {e}")
        return False
