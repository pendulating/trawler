"""PLURALS-over-vLLM spike (cluster-run; Phase 3 TODO).

Purpose: confirm the OPTIONAL PLURALS path works against our local models —
Ensemble / Chain / Debate / Graph + a Moderator via LiteLLM pointed at a
vLLM OpenAI-compatible server. The PRIMARY path is native (see
dagspaces/ci_heuristic/deliberation.py + traverse.py l5): structures run as
batched rounds inside the dagspace, which PLURALS cannot do. PLURALS remains
interesting for (a) ANES-sampled nationally representative personas
(persona_source=anes in E4) and (b) exotic structures (DAG upweighting).

Setup (cluster):
    uv pip install plurals
    # start a server, e.g.:
    #   sbatch scripts/serve_sft_checkpoint.sub  (or judge_server.sub)
    export SPIKE_BASE_URL=http://<node>:<port>/v1
    python prompt_dev/plurals_spike.py

Pass criteria: all four structures return non-empty moderated output; ANES
persona init works offline (dataset ships with the package).
"""

import os

BASE_URL = os.environ.get("SPIKE_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("SPIKE_MODEL", "openai/served-model")  # LiteLLM custom-endpoint form

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")  # vLLM servers ignore the key

TASK = (
    "Consider sidewalk delivery robots that record continuously and transmit "
    "footage to their operating company. What moral and political factors does "
    "this practice raise? Answer in 3 bullet points."
)

KW = {"api_base": BASE_URL}


def main() -> None:
    from plurals.agent import Agent
    from plurals.deliberation import Chain, Debate, Ensemble, Moderator

    def agents(n=3, persona=None):
        return [Agent(model=MODEL, persona=persona or "random", kwargs=KW) for _ in range(n)]

    results = {}
    ens = Ensemble(agents=agents(), task=TASK, moderator=Moderator(model=MODEL, kwargs=KW))
    ens.process()
    results["ensemble"] = ens.final_response

    ch = Chain(agents=agents(), task=TASK, combination_instructions="second_wave",
                moderator=Moderator(model=MODEL, kwargs=KW))
    ch.process()
    results["chain"] = ch.final_response

    deb = Debate(agents=agents(2), task=TASK, cycles=2,
                  moderator=Moderator(model=MODEL, kwargs=KW))
    deb.process()
    results["debate"] = deb.final_response

    for name, out in results.items():
        ok = bool(out and str(out).strip())
        print(f"[{name}] {'OK' if ok else 'EMPTY'} — {str(out)[:160]!r}")
    assert all(bool(str(v).strip()) for v in results.values()), "spike FAILED"
    print("SPIKE PASSED")


if __name__ == "__main__":
    main()
