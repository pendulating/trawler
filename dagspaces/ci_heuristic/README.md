# ci_heuristic — CI decision-heuristic traversal harness

Can LLMs work through Nissenbaum's 9-step Contextual Integrity decision
heuristic — not just output privacy verdicts? Part 2 of *"Tracing Norms
Around the Imaging of Public Space"* (CVPR 2027).

Design rationale: `planning/ci-heuristic-llm-experiments.md` in the umbrella
repo (changing-norms-in-public). Gold corpus + annotation guide: `corpus/`.
Canonical heuristic text: `heuristic_text.py`. Step schemas: `schemas.py`.

Sibling dagspace `vlm_geoprivacy_aug` is Part 1 (fast-judgment shifts);
E5 joins the two on shared capture-device variant ids.
