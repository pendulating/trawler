"""Tier B synthetic factorial vignette generator.

Each vignette states an entrenched informational norm for a context, then
describes a flow that either (a) conforms (`none` / prima_facie=no), (b)
departs on EXACTLY ONE parameter (prima_facie=yes, departed_parameter set),
or (c) describes a novel practice for which no entrenched norm exists
(prima_facie=incomplete_norms). Ground truth holds *by construction* —
generation is deterministic templating, seeded; no LLM in the loop.

Hard-case axes: multi-TP norms (one TP departs, one held) and sender≠subject
flows (extraction stressors, not label changers).

Usage:
    python -m dagspaces.ci_heuristic.tier_b_generator [--out corpus/tier_b/vignettes.parquet]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from dataclasses import dataclass, field

import pandas as pd

PARAMETERS = ["sender", "recipient", "subject", "information_type", "transmission_principle"]


@dataclass(frozen=True)
class ContextNorm:
    """One context with an entrenched norm and per-parameter departures.

    Field values are noun phrases that slot into the flow templates. Each
    departure value must be unambiguous: clearly outside the stated norm.
    `tp_secondary` (optional) makes this a multi-TP norm — the secondary TP
    is always held constant, so single-departure semantics are preserved.
    """

    key: str
    context: str                 # rendered context sentence fragment
    sender: str
    recipient: str
    subject: str
    information_type: str
    tp: str                      # primary transmission principle, rendered as a clause
    tp_secondary: str | None = None
    departures: dict[str, list[str]] = field(default_factory=dict)
    sender_is_subject: bool = True


CONTEXTS: list[ContextNorm] = [
    ContextNorm(
        key="health_gp",
        context="primary medical care",
        sender="the patient",
        recipient="their doctor",
        subject="the patient",
        information_type="details of their symptoms and medical history",
        tp="voluntarily, because the doctor needs it for diagnosis and treatment",
        tp_secondary="in confidence, not to be shared beyond the care team",
        departures={
            "sender": ["the patient's pharmacy app, automatically and without being asked",
                        "the patient's employer"],
            "recipient": ["the marketing department of an insurance company",
                           "the patient's landlord"],
            "subject": ["the patient's sibling, who is not under this doctor's care"],
            "information_type": ["the patient's month-by-month location history",
                                   "the patient's social media activity"],
            "transmission_principle": ["under a mandatory upload the patient cannot decline",
                                         "in exchange for the doctor selling it to advertisers"],
        },
    ),
    ContextNorm(
        key="education_teacher",
        context="school education",
        sender="a student",
        recipient="their teacher",
        subject="the student",
        information_type="an essay submitted for the class assignment",
        tp="for the purpose of assessment and feedback",
        departures={
            "sender": ["the student's tutoring app, without the student's involvement"],
            "recipient": ["an online advertising network",
                           "a college admissions data broker"],
            "subject": ["the student's parents' finances"],
            "information_type": ["a webcam recording of the student's bedroom study sessions",
                                   "the student's private group-chat messages"],
            "transmission_principle": ["under a requirement that it be posted publicly with the student's name",
                                         "in exchange for the teacher receiving a commission from a publisher"],
        },
    ),
    ContextNorm(
        key="workplace_hr",
        context="workplace human resources",
        sender="an employee",
        recipient="the HR department",
        subject="the employee",
        information_type="their bank details and tax forms for payroll",
        tp="as required to administer pay and benefits",
        tp_secondary="held confidentially within HR",
        departures={
            "sender": ["the employee's personal banking app, syncing automatically"],
            "recipient": ["the employee's direct teammates",
                           "a third-party recruitment firm"],
            "subject": ["the employee's spouse"],
            "information_type": ["the employee's off-hours dating profile activity",
                                   "the employee's personal medical appointments"],
            "transmission_principle": ["published in the company's internal newsletter",
                                         "sold to financial product marketers"],
        },
    ),
    ContextNorm(
        key="banking",
        context="retail banking",
        sender="a customer",
        recipient="their bank",
        subject="the customer",
        information_type="transaction details needed to process a disputed charge",
        tp="at the customer's request, to resolve the dispute",
        departures={
            "sender": ["a data broker who purchased the customer's records"],
            "recipient": ["a travel marketing agency",
                           "the customer's ex-partner"],
            "subject": ["the customer's teenage child"],
            "information_type": ["the customer's full web browsing history"],
            "transmission_principle": ["without the customer's knowledge, for resale",
                                         "as a condition of keeping the account open, with no opt-out"],
        },
    ),
    ContextNorm(
        key="friendship_confidence",
        context="a close friendship",
        sender="one friend",
        recipient="another close friend",
        subject="the first friend",
        information_type="details of their recent breakup",
        tp="in confidence, expecting it stays between them",
        departures={
            "sender": ["the friend's smart speaker, which recorded the conversation"],
            "recipient": ["the first friend's coworkers",
                           "a public gossip account"],
            "subject": ["the ex-partner's new relationship"],
            "information_type": ["the first friend's therapy notes"],
            "transmission_principle": ["posted publicly online for engagement",
                                         "traded to mutual acquaintances for other gossip"],
        },
        sender_is_subject=True,
    ),
    ContextNorm(
        key="library_circulation",
        context="public library services",
        sender="a library patron",
        recipient="the circulation librarian",
        subject="the patron",
        information_type="the titles they are borrowing",
        tp="as needed to manage loans and returns",
        tp_secondary="kept confidential under librarian professional ethics",
        departures={
            "sender": ["the patron's e-reader vendor"],
            "recipient": ["local law enforcement, absent any warrant",
                           "a political campaign"],
            "subject": ["the patron's book club members"],
            "information_type": ["the patron's home address and daily visit times"],
            "transmission_principle": ["published on a public list of who reads what",
                                         "sold to publishers for targeted marketing"],
        },
    ),
    ContextNorm(
        key="clergy_confession",
        context="religious pastoral care",
        sender="a congregant",
        recipient="their clergy member",
        subject="the congregant",
        information_type="a personal moral struggle disclosed in confidence",
        tp="under an expectation of absolute confidentiality",
        departures={
            "sender": ["the congregation's livestream system, recording without notice"],
            "recipient": ["the congregation's newsletter editor",
                           "the congregant's family members"],
            "subject": ["another congregant's marriage"],
            "information_type": ["the congregant's donation amounts and financial records"],
            "transmission_principle": ["shared as a sermon example identifying the congregant",
                                         "reported to the congregant's employer"],
        },
    ),
    ContextNorm(
        key="voting",
        context="democratic elections",
        sender="a voter",
        recipient="the election authority",
        subject="the voter",
        information_type="their marked ballot",
        tp="under ballot secrecy, counted but never linked back to them",
        departures={
            "sender": ["the voter's party precinct captain, submitting on their behalf without authorization"],
            "recipient": ["the voter's union representative",
                           "a campaign analytics firm"],
            "subject": ["the voter's household members' choices"],
            "information_type": ["the voter's name attached to their specific choices"],
            "transmission_principle": ["published in a name-and-vote registry",
                                         "disclosed to employers on request"],
        },
    ),
    ContextNorm(
        key="commerce_returns",
        context="retail commerce",
        sender="a shopper",
        recipient="the store's returns desk",
        subject="the shopper",
        information_type="the receipt and item details for a return",
        tp="to verify the purchase and process the refund",
        departures={
            "sender": ["the shopper's credit card network, pushing full purchase histories"],
            "recipient": ["a health insurance underwriter",
                           "the shopper's neighborhood association"],
            "subject": ["the gift's original giver and their spending habits"],
            "information_type": ["the shopper's in-store movement tracked by their phone"],
            "transmission_principle": ["retained indefinitely and matched to their face for store-wide tracking",
                                         "sold to data brokers as a condition of the refund"],
        },
    ),
    ContextNorm(
        key="tax_filing",
        context="tax administration",
        sender="a taxpayer's accountant",
        recipient="the national tax authority",
        subject="the taxpayer",
        information_type="the income figures required on the annual return",
        tp="as legally mandated for tax assessment",
        tp_secondary="protected by taxpayer confidentiality rules",
        departures={
            "sender": ["the taxpayer's bank, volunteering records unprompted"],
            "recipient": ["journalists at a local newspaper",
                           "the taxpayer's competitors"],
            "subject": ["the taxpayer's clients and their finances"],
            "information_type": ["the taxpayer's medical spending itemized by condition"],
            "transmission_principle": ["published in a searchable public database",
                                         "shared with private debt collectors for a fee"],
        },
        sender_is_subject=False,
    ),
]


# Novel practices with NO entrenched norm — correct s6 answer: incomplete_norms.
INCOMPLETE_PRACTICES = [
    ("ar_identity_overlay", "augmented-reality glasses that overlay the names and public profiles of strangers the wearer passes on the street, a capability that did not previously exist"),
    ("bci_mood_stream", "a brain-computer gaming interface that streams the player's real-time emotional state to the game's difficulty engine, a kind of signal no prior practice governed"),
    ("robot_garden_lidar", "sidewalk delivery robots that build centimeter-scale 3D maps of the private front gardens they pass, a data product with no pre-existing social practice"),
    ("dream_journal_ai", "a sleep-tracking earbud that transcribes fragments of sleep-talking and offers them to the user's own AI assistant for personality tuning"),
    ("crowd_gait_prediction", "public-transit cameras that infer commuters' likely destinations from gait and posture before any journey is chosen"),
    ("ambient_stress_pricing", "cafe seating sensors that estimate customer stress levels from posture and adjust per-table pricing in real time"),
]


FLOW_TEMPLATES = [
    "In the context of {context}, {sender} conveys {info} about {subject} to {recipient}, {tp}.",
    "Within {context}, {info} about {subject} flows from {sender} to {recipient}, {tp}.",
    "In {context}, {recipient} receives {info} about {subject} from {sender}, {tp}.",
]

NORM_TEMPLATES = [
    "A long-settled expectation in {context} is that {sender} shares {info} about {subject} with {recipient}, {tp}{tp2}.",
    "It is an entrenched norm of {context} that {info} about {subject} flows from {sender} to {recipient}, {tp}{tp2}.",
]

PRACTICE_LEAD = [
    "Now consider the following practice:",
    "A new practice has been observed:",
    "Assess this practice:",
]


def _render_norm(ctx: ContextNorm, rng: random.Random) -> str:
    tp2 = f", and {ctx.tp_secondary}" if ctx.tp_secondary else ""
    tmpl = rng.choice(NORM_TEMPLATES)
    return tmpl.format(context=ctx.context, sender=ctx.sender, info=ctx.information_type,
                        subject=ctx.subject, recipient=ctx.recipient, tp=ctx.tp, tp2=tp2)


def _render_flow(ctx: ContextNorm, values: dict[str, str], rng: random.Random) -> str:
    tp = values["transmission_principle"]
    if ctx.tp_secondary:
        tp = f"{tp}, and {ctx.tp_secondary}"
    tmpl = rng.choice(FLOW_TEMPLATES)
    return tmpl.format(context=ctx.context, sender=values["sender"], info=values["information_type"],
                        subject=values["subject"], recipient=values["recipient"], tp=tp)


def _case_id(*parts: str) -> str:
    slug = "__".join(parts)
    digest = hashlib.sha1(slug.encode()).hexdigest()[:8]
    return f"tier_b_{slug[:60]}_{digest}"


def generate(seed: int = 7, phrasings_per_cell: int = 2) -> pd.DataFrame:
    """Generate the full Tier B corpus. Deterministic for a given seed."""
    rng = random.Random(seed)
    rows: list[dict] = []

    for ctx in CONTEXTS:
        norm_values = {p: getattr(ctx, p if p != "transmission_principle" else "tp") for p in PARAMETERS}

        # Single-parameter departures (prima_facie = yes)
        for param in PARAMETERS:
            for d_i, departing in enumerate(ctx.departures.get(param, [])):
                for v in range(phrasings_per_cell):
                    values = dict(norm_values)
                    values[param] = departing
                    norm_txt, flow_txt = _render_norm(ctx, rng), _render_flow(ctx, values, rng)
                    rows.append({
                        "case_id": _case_id(ctx.key, param, str(d_i), str(v)),
                        "context_key": ctx.key,
                        "practice_input": f"{norm_txt}\n\n{rng.choice(PRACTICE_LEAD)} {flow_txt}",
                        "departed_parameter": param,
                        "prima_facie": "yes",
                        "multi_tp": ctx.tp_secondary is not None,
                        "sender_is_subject": ctx.sender_is_subject,
                        "norm_statement": norm_txt,
                        "flow_statement": flow_txt,
                        "gold_values": str({**values, "_norm": norm_values}),
                    })

        # No-departure controls (prima_facie = no)
        for v in range(phrasings_per_cell):
            norm_txt, flow_txt = _render_norm(ctx, rng), _render_flow(ctx, norm_values, rng)
            rows.append({
                "case_id": _case_id(ctx.key, "none", "0", str(v)),
                "context_key": ctx.key,
                "practice_input": f"{norm_txt}\n\n{rng.choice(PRACTICE_LEAD)} {flow_txt}",
                "departed_parameter": "none",
                "prima_facie": "no",
                "multi_tp": ctx.tp_secondary is not None,
                "sender_is_subject": ctx.sender_is_subject,
                "norm_statement": norm_txt,
                "flow_statement": flow_txt,
                "gold_values": str({**norm_values, "_norm": norm_values}),
            })

    # Norm-incomplete practices (prima_facie = incomplete_norms)
    for key, practice in INCOMPLETE_PRACTICES:
        for v in range(phrasings_per_cell):
            lead = rng.choice(PRACTICE_LEAD)
            rows.append({
                "case_id": _case_id("incomplete", key, "0", str(v)),
                "context_key": f"incomplete_{key}",
                "practice_input": f"{lead} {practice}. Assess whether this practice violates entrenched informational norms.",
                "departed_parameter": "none",
                "prima_facie": "incomplete_norms",
                "multi_tp": False,
                "sender_is_subject": True,
                "norm_statement": "",
                "flow_statement": practice,
                "gold_values": "{}",
            })

    df = pd.DataFrame(rows)
    assert not df["case_id"].duplicated().any(), "case_id collision"
    return df


def validate_structure(df: pd.DataFrame) -> dict[str, int]:
    """Structural self-checks: label semantics hold by construction."""
    problems = 0
    for _, r in df.iterrows():
        if r["prima_facie"] == "yes":
            assert r["departed_parameter"] in PARAMETERS, r["case_id"]
            assert r["norm_statement"], r["case_id"]
        elif r["prima_facie"] == "no":
            assert r["departed_parameter"] == "none" and r["norm_statement"], r["case_id"]
        elif r["prima_facie"] == "incomplete_norms":
            assert not r["norm_statement"], r["case_id"]
        else:
            problems += 1
    counts = {
        "total": len(df),
        "departures": int((df["prima_facie"] == "yes").sum()),
        "controls": int((df["prima_facie"] == "no").sum()),
        "incomplete": int((df["prima_facie"] == "incomplete_norms").sum()),
        "multi_tp": int(df["multi_tp"].sum()),
        "sender_ne_subject": int((~df["sender_is_subject"]).sum()),
        "problems": problems,
    }
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    default_out = os.path.join(os.path.dirname(__file__), "corpus", "tier_b", "vignettes.parquet")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--phrasings", type=int, default=4)
    args = parser.parse_args()

    df = generate(seed=args.seed, phrasings_per_cell=args.phrasings)
    counts = validate_structure(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}: {counts}")
    print(df.groupby(["prima_facie", "departed_parameter"]).size())


if __name__ == "__main__":
    main()
