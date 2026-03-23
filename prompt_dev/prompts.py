from __future__ import annotations


def theory_full_name(theory: str) -> str:
    value = (theory or "").strip().upper()
    if value == "CI":
        return "Contextual Integrity"
    if value == "IG":
        return "Institutional Grammar"
    return theory


def p_elic(theory: str) -> str:
    return f"Tell me everything you know about {theory_full_name(theory)}."


def p_feedback(text: str) -> str:
    return (
        "You are reviewing a theoretical summary for factuality and coherence.\n"
        "Identify logical inconsistencies, hallucinations, omissions, and ambiguous phrasing.\n"
        "Return concise feedback bullets.\n\n"
        f"Summary to review:\n{text}"
    )


def p_refine(previous_text: str, feedback_text: str, theory: str) -> str:
    full_theory = theory_full_name(theory)
    return (
        f"Revise this {full_theory} summary using the feedback.\n"
        "Produce a cleaner, more precise version with stronger theoretical grounding.\n\n"
        f"Previous summary:\n{previous_text}\n\n"
        f"Feedback:\n{feedback_text}"
    )


def p_decompose(overview_text: str, theory: str) -> str:
    full_theory = theory_full_name(theory)
    return (
        f"Given this overview of {full_theory}, decompose it into core sub-components.\n"
        "Return strict JSON with this schema:\n"
        '{"components":[{"name":"string","summary":"string"}]}\n\n'
        f"Overview:\n{overview_text}"
    )


def p_define(component_name: str, component_summary: str, overview_text: str, theory: str) -> str:
    full_theory = theory_full_name(theory)
    return (
        f"Provide a rigorous definition for the {full_theory} component '{component_name}'.\n"
        "Include: (1) formal definition, (2) contrast with nearby concepts, (3) at least two examples.\n"
        "Use clear headings.\n\n"
        f"Component summary:\n{component_summary}\n\n"
        f"Theory overview context:\n{overview_text}"
    )


def p_question_gen(k_star_text: str, theory: str) -> str:
    full_theory = theory_full_name(theory)
    return (
        f"Generate 50 intermediate reasoning questions for {full_theory} based on the gold-standard context.\n"
        "Questions should stress component interactions and edge-cases.\n"
        "Return strict JSON with schema:\n"
        '{"questions":[{"id":"q1","question":"...","rationale":"..."}]}\n\n'
        f"Gold context:\n{k_star_text}"
    )


def p_cot(k_star_text: str, exemplars_text: str, question: str, theory: str) -> str:
    full_theory = theory_full_name(theory)
    return (
        f"Answer the question about {full_theory} using step-by-step reasoning grounded in verified definitions.\n"
        "First map question terms to framework definitions, then reason, then conclude.\n"
        "Format:\n"
        "Reasoning:\n...\n\nFinal Answer:\n...\n\n"
        f"Gold context K*:\n{k_star_text}\n\n"
        f"Exemplars E_k:\n{exemplars_text}\n\n"
        f"Question:\n{question}"
    )

