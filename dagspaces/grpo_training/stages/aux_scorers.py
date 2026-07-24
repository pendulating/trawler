"""Production `R-GROUND` / `R-CONTRAST` auxiliary scorers for the m-series stack.

The modular reward core (:mod:`modular_reward`) is judge-free and unit-testable:
the two listwise-judge auxiliaries are supplied as *injected callables* with the
contract ``scorer(*, completions, prompts, metadata_list) -> Sequence[float]``
(``None`` / wrong-length ⇒ the caller applies the group-neutral 0.5 fallback and
increments ``reward/<aux>/judge_failed_group_frac``). This file builds the two
production adapters and is wired in by
:func:`modular_reward._make_aux_scorers`; every unit test injects its own mocks
and never touches this module's factory.

Design (wiki/grpo_redesign/reward-ground.md, reward-contrast.md):

* **R-GROUND** — one *listwise* judge call per group against the chunk's **own
  book's** norm universe. The judge ranks the G completions by grounding and
  emits a per-candidate absolute grounding score; the two are blended into a
  scalar ``s_i = w_r·(n−rank_i)/(n−1) + (1−w_r)·grounding_i`` (``rank_weight``
  ``w_r = 0.5``), exactly the v9 lineage's correct-universe pass. The rubric is
  **slimmed to grounding only** (norm awareness + flow governance); the
  appropriateness criterion is deleted — direction is scored verifiably by
  R-OUTCOME, and keeping it here would double-count it.

* **R-CONTRAST** — score the completion's flows against ONE **wrong book's**
  universe (retrieval + absolute grounding, **no ranking**) and reward the
  complement ``r_contrast = 1 − grounding_wrong``. The wrong book is sampled
  uniformly over the *other* training books, **seeded by ``chunk_id``** — fixed
  per prompt across the whole run so all G completions face the same wrong book
  and the term enters the prescreen cache signature deterministically. High
  reward = the extraction is distinctively *of its book*, not generic privacy
  boilerplate that grounds equally well anywhere.

Reuse, never edit (the parallel-stack rule, wiki/grpo_redesign/migration.md):
this module *imports* the keeper's frozen judge/retrieval plumbing —
:class:`~.clients.EmbeddingClient` / :class:`~.clients.JudgeClient` /
:class:`~.clients.NormRetriever`, the ``_flow_to_query`` / ``_rankings_to_scores``
helpers from :mod:`.online_rground`, and the ``CompletionRankingJudgment``
schema — and edits none of them. Both auxiliaries reuse the keeper's listwise
``judge_ranking_batch`` transport: R-GROUND consumes rank+grounding, R-CONTRAST
consumes **only** the absolute grounding score (rank discarded — ranks are
meaningless across universes; this was already the v9 wrong-pass convention),
and both get a clean ``None``-on-failure signal from the client.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dagspaces.common.json_extraction import extract_json_from_text

from .clients import EmbeddingClient, JudgeClient, NormRetriever
from .online_rground import _flow_to_query, _flatten_flow, _rankings_to_scores

__all__ = ["make_aux_scorers", "seeded_wrong_book"]


# ---------------------------------------------------------------------------
# Grounding-only judge prompt (rubric slimmed vs the keeper's reward_judge_
# ranking.yaml: the third "APPROPRIATENESS CONSISTENCY" criterion is DELETED —
# reward-ground.md "Judge rubric — slimmed to grounding only"). The keeper's
# CompletionRankingJudgment schema is reused verbatim (candidate_index / rank /
# grounding_score); only the *instruction* changes, so no keeper edit is needed.
# Both the own-universe (ground) and wrong-universe (contrast) calls use this
# same prompt — the judge grounds candidates against whatever norm set it is
# handed and does not know which universe it is.
# ---------------------------------------------------------------------------
GROUND_RANKING_SYSTEM = (
    "You are an expert in Helen Nissenbaum's Contextual Integrity framework. "
    "You compare several candidate analyses of the SAME source passage and rank "
    "them by how well they are grounded in a provided set of norms from the "
    "passage's normative universe.\n\n"
    "Each candidate extracts one or more information flows — transfers of "
    "information between agents — annotated with invoked norms.\n\n"
    "Ranking criteria, in order of importance:\n"
    "1. NORM AWARENESS: do the candidate's invoked norms semantically match the "
    "provided norms? Semantic equivalence suffices; exact wording is not "
    "required.\n"
    "2. FLOW GOVERNANCE: are the candidate's extracted flows actually governed "
    "by the provided norms — do the norms regulate information flows of that "
    "type, between those kinds of actors, about that kind of information, in "
    "that context?\n\n"
    "Judge ONLY grounding (norm awareness + flow governance). Do NOT reward or "
    "penalise the candidate's appropriateness verdict — that is scored "
    "elsewhere.\n\n"
    "Produce a STRICT ranking — every candidate gets a distinct rank, no ties. "
    "Also give each candidate an absolute grounding_score from 0.0 (ungrounded) "
    "to 1.0 (fully grounded). Respond with a JSON object matching the required "
    "schema, with exactly one rankings entry per candidate."
)

GROUND_RANKING_TEMPLATE = (
    "## Source Text\n{{chunk_text}}\n\n"
    "## Retrieved Norms (from the passage's normative universe)\n"
    "{{norm_universe_json}}\n\n"
    "## Candidates\n{{candidates_block}}\n\n"
    "## Evaluation\n"
    "Rank all {{n_candidates}} candidates from best grounded (rank 1) to worst "
    "grounded (rank {{n_candidates}}) in the retrieved norms, applying norm "
    "awareness then flow governance. Assign each candidate an absolute "
    "grounding_score in [0.0, 1.0]. Ranks must be distinct.\n\n"
    "Provide your evaluation as a JSON object."
)


def _ranking_json_schema() -> dict[str, Any]:
    """The keeper's ``CompletionRankingJudgment`` JSON schema (reused, not
    edited). Imported lazily so this module has no hard pydantic dependency at
    import time (the unit tests inject a mock judge and never call this)."""
    from ..schemas import CompletionRankingJudgment

    return CompletionRankingJudgment.model_json_schema()


# ---------------------------------------------------------------------------
# Seeded wrong-book sampling (reward-contrast.md "Wrong-universe sampling")
# ---------------------------------------------------------------------------
def seeded_wrong_book(
    chunk_id: Any, source_id: str, all_source_ids: Sequence[str]
) -> str | None:
    """Deterministically pick ONE wrong book for a chunk.

    Uniform over the *other* training books (never ``source_id``), **seeded by
    ``chunk_id``** so the pairing is fixed per prompt across the entire run —
    every G completion in every epoch faces the same wrong book, and the term
    enters the prescreen cache signature deterministically (the v9 lineage
    resampled per call; determinism costs nothing and buys reproducibility).

    Returns ``None`` when there is no other book (a single-book universe — no
    contrast is possible), which the caller routes to the group-neutral
    fallback.
    """
    candidates = sorted(str(s) for s in all_source_ids if str(s) != str(source_id))
    if not candidates:
        return None
    # SHA1 of the chunk id → a stable index into the sorted candidate list.
    # Sorted so the mapping is independent of dict/iteration order.
    h = int(hashlib.sha1(str(chunk_id).encode("utf-8")).hexdigest(), 16)
    return candidates[h % len(candidates)]


# ---------------------------------------------------------------------------
# Shared per-group preparation (parse → queries → group embedding → block)
# ---------------------------------------------------------------------------
def _parse_flows(text: str) -> list[dict]:
    """Extract the ``flows`` list from a completion's JSON (best-effort).

    Completions reaching an auxiliary have already passed the R-VALID gate, so
    the object parses and carries a ``flows`` list; ``repair=False`` keeps a
    broken completion from being salvaged into a spurious flow set. Non-dict
    flow entries are dropped.
    """
    obj, _err = extract_json_from_text(text or "", repair=False)
    if not isinstance(obj, dict):
        return []
    flows = obj.get("flows")
    if not isinstance(flows, list):
        return []
    return [f for f in flows if isinstance(f, dict)]


def _candidate_block(flows_by_completion: list[list[dict]]) -> tuple[str, list[str]]:
    """Build the judge's ``### Candidate {i}`` block + aligned per-candidate list.

    Mirrors OnlineRGround's block construction so the keeper's judge/reranker
    clients parse it unchanged. Candidate ``i`` is completion ``i`` (order
    preserved), which is the alignment ``_rankings_to_scores`` assumes.
    """
    candidate_texts: list[str] = []
    for flows in flows_by_completion:
        if flows:
            candidate_texts.append(json.dumps(flows, ensure_ascii=False, indent=1))
        else:
            candidate_texts.append(
                "This candidate declares the passage contains "
                "NO information flows."
            )
    block = "\n\n".join(
        f"### Candidate {i}\n{txt}" for i, txt in enumerate(candidate_texts)
    )
    return block, candidate_texts


def _group_embedding(
    embedding_client: EmbeddingClient, flows_by_completion: list[list[dict]]
) -> np.ndarray | None:
    """One shared group embedding: mean of every flow's retrieval-query embedding.

    A single retrieval per group keeps the evidence identical across candidates
    (the listwise-judge requirement). Returns ``None`` when no completion
    carries a flow (nothing to embed) or the encode call degenerates to zeros —
    both route the group to the neutral fallback rather than judging against
    garbage norms.
    """
    queries = [
        _flow_to_query(_flatten_flow(f))
        for flows in flows_by_completion
        for f in flows
    ]
    if not queries:
        return None
    embeddings = embedding_client.encode_batch(queries)
    embeddings = np.asarray(embeddings)
    if embeddings.size == 0:
        return None
    group_emb = np.mean(embeddings, axis=0)
    norm = float(np.linalg.norm(group_emb))
    if norm <= 0.0 or not np.asarray(group_emb).any():
        return None
    return group_emb / norm


# ---------------------------------------------------------------------------
# The two scorers
# ---------------------------------------------------------------------------
class _GroundScorer:
    """R-GROUND: listwise grounding against the chunk's OWN book's universe.

    One ``judge_ranking_batch`` call per group; rank blended with the absolute
    grounding score per candidate (``rank_weight`` w_r). Judge failure (the
    client returns ``None``) → the scorer returns ``None`` so the modular core
    applies the uniform 0.5 group-neutral fallback and marks the group failed.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        judge_client: JudgeClient,
        norm_retriever: NormRetriever,
        *,
        rank_top_k: int = 5,
        rank_weight: float = 0.5,
        system_prompt: str = GROUND_RANKING_SYSTEM,
        prompt_template: str = GROUND_RANKING_TEMPLATE,
        json_schema: dict[str, Any] | None = None,
    ):
        self.embedding_client = embedding_client
        self.judge_client = judge_client
        self.norm_retriever = norm_retriever
        self.rank_top_k = int(rank_top_k)
        self.rank_weight = float(rank_weight)
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.json_schema = json_schema

    def __call__(
        self,
        *,
        completions: Sequence[str],
        prompts: Sequence[str],
        metadata_list: Sequence[Mapping[str, Any]],
    ) -> list[float] | None:
        n = len(completions)
        if n == 0:
            return []
        flows_by_completion = [_parse_flows(c) for c in completions]
        group_emb = _group_embedding(self.embedding_client, flows_by_completion)
        if group_emb is None:
            return None  # nothing to retrieve on → group-neutral

        meta0 = metadata_list[0] if metadata_list else {}
        source_id = str(meta0.get("source_id", ""))
        chunk_text = str(meta0.get("chunk_text", ""))

        norms_json, _sims = self.norm_retriever.retrieve(
            group_emb, source_id, return_scores=True, top_k=self.rank_top_k
        )
        block, candidate_list = _candidate_block(flows_by_completion)
        item = {
            "chunk_text": chunk_text,
            "norm_universe_json": norms_json,
            "candidates_block": block,
            "candidates": candidate_list,
            "n_candidates": n,
        }
        results = self.judge_client.judge_ranking_batch(
            [item],
            system_prompt=self.system_prompt,
            prompt_template=self.prompt_template,
            json_schema=self.json_schema,
        )
        rankings = results[0] if results else None
        if rankings is None:
            return None  # judge failed → group-neutral
        return _rankings_to_scores(rankings, n, rank_weight=self.rank_weight)


class _ContrastScorer:
    """R-CONTRAST: ``1 − grounding`` against ONE seeded WRONG book's universe.

    The wrong book is fixed per ``chunk_id`` (:func:`seeded_wrong_book`). One
    ``judge_ranking_batch`` call against the wrong universe, from which only the
    absolute grounding score is read — **ranks are discarded** (meaningless
    across universes; the v9 wrong-pass convention). ``r_contrast = 1 −
    grounding_wrong``: high reward = the extraction grounds *poorly* in a random
    other book, i.e. it is distinctively of its own book. Judge failure or a
    single-book universe → ``None`` (group-neutral).
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        judge_client: JudgeClient,
        norm_retriever: NormRetriever,
        all_source_ids: Sequence[str],
        *,
        rank_top_k: int = 5,
        system_prompt: str = GROUND_RANKING_SYSTEM,
        prompt_template: str = GROUND_RANKING_TEMPLATE,
        json_schema: dict[str, Any] | None = None,
    ):
        self.embedding_client = embedding_client
        self.judge_client = judge_client
        self.norm_retriever = norm_retriever
        self.all_source_ids = [str(s) for s in all_source_ids]
        self.rank_top_k = int(rank_top_k)
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.json_schema = json_schema

    def __call__(
        self,
        *,
        completions: Sequence[str],
        prompts: Sequence[str],
        metadata_list: Sequence[Mapping[str, Any]],
    ) -> list[float] | None:
        n = len(completions)
        if n == 0:
            return []
        meta0 = metadata_list[0] if metadata_list else {}
        source_id = str(meta0.get("source_id", ""))
        chunk_id = meta0.get("chunk_id", "")
        chunk_text = str(meta0.get("chunk_text", ""))

        wrong_book = seeded_wrong_book(chunk_id, source_id, self.all_source_ids)
        if wrong_book is None:
            return None  # single-book universe → no contrast possible

        flows_by_completion = [_parse_flows(c) for c in completions]
        group_emb = _group_embedding(self.embedding_client, flows_by_completion)
        if group_emb is None:
            return None

        wrong_norms, _sims = self.norm_retriever.retrieve(
            group_emb,
            source_id,
            contrastive_source=wrong_book,
            return_scores=True,
            top_k=self.rank_top_k,
        )
        block, candidate_list = _candidate_block(flows_by_completion)
        item = {
            "chunk_text": chunk_text,
            "norm_universe_json": wrong_norms,
            "candidates_block": block,
            "candidates": candidate_list,
            "n_candidates": n,
        }
        results = self.judge_client.judge_ranking_batch(
            [item],
            system_prompt=self.system_prompt,
            prompt_template=self.prompt_template,
            json_schema=self.json_schema,
        )
        rankings = results[0] if results else None
        if rankings is None:
            return None  # judge failed → group-neutral

        # Read the ABSOLUTE grounding score only (rank discarded).
        wrong_grounding = [0.0] * n
        for entry in rankings:
            idx = entry.get("candidate_index", -1)
            if 0 <= idx < n:
                wrong_grounding[idx] = max(
                    0.0, min(1.0, float(entry.get("grounding_score", 0.0)))
                )
        return [1.0 - g for g in wrong_grounding]


# ---------------------------------------------------------------------------
# Factory (config → the two callables; reuses the keeper's client plumbing)
# ---------------------------------------------------------------------------
def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict-like or attr-like config node (mirrors
    answerer_client._cfg_get so config access is uniform across m-series code)."""
    if node is None:
        return default
    if isinstance(node, Mapping):
        val = node.get(key, default)
    elif hasattr(node, "get") and not isinstance(node, (str, bytes)):
        try:
            val = node.get(key, default)
        except Exception:
            val = getattr(node, key, default)
    else:
        val = getattr(node, key, default)
    return default if val is None else val


def _resolve_judge_url(grpo_cfg: Any) -> str:
    """Resolve the aux judge URL — the SHARED gemma-4-31b server, same plumbing
    as the answerer. Order: an explicit ``aux_judge``/``answerer`` ``base_url_env``
    env var → an explicit ``base_url`` → ``VLLM_SERVER_URL`` → localhost default.
    """
    node = _cfg_get(grpo_cfg, "aux_judge", None) or _cfg_get(grpo_cfg, "answerer", None)
    base_url_env = str(_cfg_get(node, "base_url_env", "VLLM_SERVER_URL"))
    return (
        os.environ.get(base_url_env, "")
        or str(_cfg_get(node, "base_url", "") or "")
        or os.environ.get("VLLM_SERVER_URL", "")
        or "http://localhost:8000"
    )


def make_aux_scorers(
    cfg: Any,
    grpo_cfg: Any,
    norm_universes: dict[str, list],
    active_aux: Sequence[str],
    *,
    embeddings_dir: str = "",
    embedding_client: EmbeddingClient | None = None,
    judge_client: JudgeClient | None = None,
    norm_retriever: NormRetriever | None = None,
):
    """Build the ``(ground_scorer, contrast_scorer)`` pair for the active auxiliaries.

    The judge is **gemma-4-31b on the shared vLLM server** (same URL resolution
    as the answerer). Retrieval reuses the keeper's ``EmbeddingClient`` +
    ``NormRetriever`` (Qwen3-Embedding-8B). Each returned entry is ``None`` when
    its auxiliary is not active, so the caller can unpack unconditionally.

    Injection: tests (and any caller that already holds live clients) pass
    ``embedding_client`` / ``judge_client`` / ``norm_retriever`` and this factory
    builds none of its own — the only path exercised on CPU. When they are
    omitted the production clients are constructed from ``cfg``/``grpo_cfg``;
    that path needs the live embedding + judge servers and is Phase-B smoke.
    """
    active = set(str(a) for a in active_aux)
    want_ground = "ground" in active
    want_contrast = "contrast" in active
    if not (want_ground or want_contrast):
        return None, None

    all_source_ids = sorted(str(s) for s in norm_universes.keys())
    rank_top_k = int(_cfg_get(grpo_cfg, "rank_top_k", 5))
    rank_weight = float(_cfg_get(grpo_cfg, "rank_weight", 0.5))

    # Build production clients only for what the injection did not supply. All
    # three are needed once any auxiliary is active.
    if embedding_client is None:
        emb_port = int(_cfg_get(grpo_cfg, "embedding_server_port", 8001))
        embedding_url = (
            str(_cfg_get(grpo_cfg, "embedding_server_url", "") or "")
            or os.environ.get("GRPO_EMBEDDING_SERVER_URL", "")
            or f"http://localhost:{emb_port}"
        )
        emb_model_name = ""
        try:
            from omegaconf import OmegaConf

            emb_model_name = str(
                OmegaConf.select(cfg, "embedding_model.model_source", default=None)
                or OmegaConf.select(cfg, "model.embedding_model_source", default=None)
                or ""
            )
        except Exception:
            pass
        embedding_client = EmbeddingClient(
            base_url=embedding_url, model_name=emb_model_name or "default"
        )

    if judge_client is None:
        judge_url = _resolve_judge_url(grpo_cfg)
        node = _cfg_get(grpo_cfg, "aux_judge", None) or _cfg_get(grpo_cfg, "answerer", None)
        judge_model = str(_cfg_get(node, "model", "gemma-4-31b"))
        judge_client = JudgeClient(
            base_url=judge_url,
            model_name=judge_model,
            max_workers=int(_cfg_get(grpo_cfg, "judge_max_workers", 16)),
        )

    if norm_retriever is None:
        norm_retriever = NormRetriever(
            norm_universes=norm_universes,
            embeddings_dir=embeddings_dir or str(_cfg_get(grpo_cfg, "embeddings_dir", "") or ""),
            embedding_client=embedding_client,
            top_k=rank_top_k,
        )

    schema = None
    try:
        schema = _ranking_json_schema()
    except Exception:
        # Guided-decoding schema is best-effort; the judge prompt still asks for
        # the rankings JSON. Tests inject a mock judge that ignores it entirely.
        schema = None

    ground_scorer = None
    contrast_scorer = None
    if want_ground:
        ground_scorer = _GroundScorer(
            embedding_client,
            judge_client,
            norm_retriever,
            rank_top_k=rank_top_k,
            rank_weight=rank_weight,
            json_schema=schema,
        )
    if want_contrast:
        contrast_scorer = _ContrastScorer(
            embedding_client,
            judge_client,
            norm_retriever,
            all_source_ids,
            rank_top_k=rank_top_k,
            json_schema=schema,
        )
    return ground_scorer, contrast_scorer
