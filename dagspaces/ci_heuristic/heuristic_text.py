"""Canonical text of the augmented CI decision heuristic.

Step texts follow Nissenbaum (2010, p.181f) as transcribed in the paper's
methods notes; goals and guiding questions follow Kumar, Zimmer & Vitak
(2024), Table 1. These strings are the single source of truth for prompt
construction — edit here, not in prompts.
"""

from __future__ import annotations

# (step_key, step_text, goal, guiding_questions)
HEURISTIC_STEPS = {
    "s1": {
        "title": "Describe the new practice in terms of information flows",
        "text": "Describe the new practice in terms of information flows.",
        "goal": "To define the object of study in the language of CI.",
        "guiding_questions": [
            "What are people, groups, or institutions doing?",
            "How are they doing it (e.g., what tools are they using)?",
            "How does information fit into this practice?",
            "What aspects of the practice are novel, and what are extensions of existing practices?",
        ],
        "kind": "descriptive",
    },
    "s2": {
        "title": "Identify the prevailing context",
        "text": (
            "Identify the information types, activities, and purposes involved in a given "
            "information flow and link them to a prevailing context. Establish context at a "
            "familiar level of generality (e.g., 'health care') and identify potential impacts "
            "from contexts nested within it (e.g., 'teaching hospital'). Context is a social "
            "domain, not a place, platform, or industry."
        ),
        "goal": "To embed the information flow in a social context.",
        "guiding_questions": [
            "What types of information does the flow involve?",
            "What actions are occurring in this practice?",
            "Why is this information flow happening?",
            "What are actors using the information for?",
        ],
        "kind": "descriptive",
    },
    "s3": {
        "title": "Identify information subjects, senders, and recipients",
        "text": (
            "Identify information subjects, senders, and recipients. Actors can be single "
            "individuals, multiple individuals, or collectives such as organizations."
        ),
        "goal": "To clarify who and/or what is participating in the information flow.",
        "guiding_questions": [
            "Who/what is sending information?",
            "Who/what is receiving information?",
            "Who is the information about, or to whom does the information pertain?",
        ],
        "kind": "descriptive",
    },
    "s4": {
        "title": "Identify transmission principles",
        "text": (
            "Identify transmission principles: the constraints — terms and conditions — under "
            "which information transfers ought (or ought not) to occur in this context. "
            "Transmission principles can be explicit (codified in law, policy, direct "
            "statements) or implicit (inferred from the situation)."
        ),
        "goal": "To establish the conditions that govern the information flow.",
        "guiding_questions": [
            "What requirements need to be met for this information flow to occur?",
            "In what circumstances can the information flow occur or not?",
            "What stipulations exist that dictate whether this information flow can occur?",
        ],
        "kind": "descriptive",
    },
    "s5": {
        "title": "Locate entrenched informational norms and points of departure",
        "text": (
            "Locate applicable entrenched informational norms and identify significant points "
            "of departure. Norms reflect settled, collective expectations about how information "
            "flows in this context — not individual preferences. It is possible that no norms "
            "have yet developed for the activities in question (the normative structure may be "
            "incomplete); recognize this rather than forcing a match."
        ),
        "goal": (
            "To discern expectations about how information usually flows in the context and to "
            "pinpoint what the new practice changes."
        ),
        "guiding_questions": [
            "How is information typically used and managed in this context?",
            "How do these uses of information align with the broader goals, values, or purposes of the context?",
            "What, if any, specific aspects of the information flow are altered by the practice under study?",
        ],
        "kind": "descriptive",
    },
    "s6": {
        "title": "Prima facie assessment",
        "text": (
            "Prima facie assessment: a breach of entrenched informational norms yields a prima "
            "facie judgment that contextual integrity has been violated, because presumption "
            "favors the entrenched practice. A violation of norms — not a mere change in "
            "parameters — is required; if the normative structure is incomplete for these "
            "activities, say so. A prima facie violation is not, by itself, grounds to reject "
            "the practice; that determination comes only after the following evaluation steps."
        ),
        "goal": "To determine whether the practice under study violates privacy.",
        "guiding_questions": [
            "Does the information flow align with entrenched norms of this context, as established in step 5? Why or why not?",
        ],
        "kind": "pivot",
    },
    "s7": {
        "title": "Evaluation I: moral and political factors",
        "text": (
            "Consider moral and political factors affected by the practice in question. What "
            "might be the harms, the threats to autonomy and freedom? What might be the effects "
            "on power structures, implications for justice, fairness, equality, social "
            "hierarchy, democracy, and so on?"
        ),
        "goal": "To recognize the social implications of the practice under study.",
        "guiding_questions": [
            "How does the practice under study threaten autonomy or freedom?",
            "How does the practice alter power structures or power relations?",
            "How does the practice affect equality, fairness, justice, democracy?",
        ],
        "kind": "evaluative",
    },
    "s8": {
        "title": "Evaluation II: meaning in light of contextual values and ends",
        "text": (
            "Ask how the practice directly impinges on the values, goals, and ends of the "
            "context, and consider the meaning or significance of the moral and political "
            "factors in light of those contextual values, ends, purposes, and goals. What do "
            "the harms, threats to autonomy and freedom, or perturbations in power and justice "
            "MEAN in relation to this specific context?"
        ),
        "goal": "To assess how the practice under study could affect the context.",
        "guiding_questions": [
            "How does the practice under study align with the aims or goals of the context?",
            "How do the moral/political implications of the practice advance or undermine these contextual aims?",
        ],
        "kind": "evaluative",
    },
    "s9": {
        "title": "Recommendation",
        "text": (
            "On the basis of these findings, contextual integrity recommends in favor of or "
            "against the practice under study. Privacy is not absolute: a practice that "
            "violates entrenched norms may still be acceptable if it advances contextual ends "
            "or societal goals that outweigh the breach — but the burden of proof falls on the "
            "practice's proponents."
        ),
        "goal": "To judge, based on the analysis, what the course of action should be.",
        "guiding_questions": [
            "Should the practice under study continue as is, or should it be rejected?",
            "What, if any, modifications or conditions should be implemented?",
        ],
        "kind": "prescriptive",
    },
}

# Compact rendering of the whole heuristic for L1 (heuristic-in-context) runs.
FULL_HEURISTIC_TEXT = "\n".join(
    f"Step {i + 1} — {step['title']}: {step['text']}"
    for i, step in enumerate(HEURISTIC_STEPS.values())
)

# The descriptive/prescriptive firewall, appended to steps 1-5 prompts.
FIREWALL_GUARD = (
    "IMPORTANT: This is a DESCRIPTIVE step. Do not evaluate, judge, or express "
    "approval or concern about the practice yet — no talk of violations, harms, "
    "risks, or appropriateness. Only identify and describe. Evaluation happens "
    "in later steps."
)
