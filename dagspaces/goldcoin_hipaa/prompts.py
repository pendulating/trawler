"""Prompt construction for GoldCoin HIPAA evaluation.

Ported from GoldCoin/eval/build_instruction_compliance.py,
build_instruction_applicability.py, and eval_llm.py.
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Alpaca prompt template (from eval_llm.py)
# ---------------------------------------------------------------------------

ALPACA_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n"
    "### Instruction: {instruction}\n"
    "### Input: {input}\n"
    "### Response:"
)

ALPACA_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n"
    "### Instruction: {instruction}\n"
    "### Response:"
)

# ---------------------------------------------------------------------------
# Instruction templates (from build_instruction_compliance.py / applicability.py)
# ---------------------------------------------------------------------------

COMPLIANCE_DIRECT_INSTRUCTION = (
    "Please determine whether the HIPAA Privacy Rule permits or forbids the case."
)

COMPLIANCE_COT_INSTRUCTION = (
    "Please assess the case for compliance with the HIPAA Privacy Rule through "
    "the following steps: Step 1: Annotate the eleven message characteristics "
    "[Sender, Sender Role, Recipient, Recipient Role, About, About Role, Type, "
    "Purpose, In Reply To, Consented By, Belief] about the flow of private "
    "information in the case as a list. Step 2: Identify and list all applicable "
    "HIPAA regulation IDs (e.g., 164.XXX) and their content. Step 3: Determine "
    "whether the HIPAA Privacy Rule permits or forbids the case."
)

APPLICABILITY_DIRECT_INSTRUCTION = (
    "Please determine whether the HIPAA Privacy Rule is applicable to the case."
)

APPLICABILITY_COT_INSTRUCTION = (
    "Please assess the applicability of the HIPAA Privacy Rule to the case "
    "through the following steps: Step 1: Annotate the message characteristics "
    "[Sender, Sender Role, Recipient, Recipient Role, About, About Role, Type] "
    "about the flow of private information in the case as a list. Step 2: "
    "Determine whether the HIPAA Privacy Rule is applicable to the case."
)

# ---------------------------------------------------------------------------
# Few-shot examples (from eval_llm.py:add_few_shot_instruction)
# ---------------------------------------------------------------------------

COMPLIANCE_FEW_SHOT = (
    "Read the case: Dr. Smith, a researcher at XYZ University, is conducting a "
    "study on the long-term effects of a specific medication on patients with a "
    "rare disease. To gather data, Dr. Smith needs access to the protected health "
    "information (PHI) of patients who have been taking the medication for the past "
    "five years. Dr. Smith's research has been approved by the institutional review "
    "board (IRB) of XYZ University.The IRB determined that the PHI necessary for "
    "Dr. Smith's research includes information about the patients' diagnoses, "
    "treatment plans, and medication history. Dr. Smith requested the necessary PHI "
    "from ABC Hospital, where most of the patients with the rare disease have been "
    "treated. The hospital's privacy board reviewed the request and found that the "
    "information needed was in line with the IRB's determination.Dr. Smith was "
    "provided with the PHI of patients who met the criteria of the study, without "
    "the need for individual patients' authorization, since the privacy board had "
    "approved the waiver for research purposes. The PHI was sent to Dr. Smith by "
    "the hospital's health information manager, ensuring that only the necessary "
    "information was shared, without any direct identifiers like names or addresses.\n"
    " ### Response: HIPAA Privacy Rule permits the case. \n"
    " ### Input: Read the case: Jane Smith, a patient at Healthy Life Clinic, had "
    "recently undergone a medical procedure and was concerned about her privacy. She "
    "requested Dr. Johnson, her primary care physician, to restrict the use and "
    "disclosure of her protected health information (PHI) related to the procedure. "
    "Dr. Johnson agreed to the restriction, and they documented the agreed-upon "
    "limitation in Jane's medical records.A few weeks later, Jane's employer, Big "
    "Corp, required her to submit medical information to their insurance provider, "
    "Insurance Co, to process her insurance claim. Jane provided written consent to "
    "Dr. Johnson's office to disclose her general medical information to Insurance "
    "Co, but reiterated the restriction on her procedure-related PHI.Dr. Johnson's "
    "office sent the necessary information to Insurance Co. However, due to a "
    "clerical error, the PHI related to Jane's procedure was included in the "
    "disclosure. Insurance Co, upon receiving the information, shared it with Big "
    "Corp as part of the claim process.Upon learning about the disclosure, Jane was "
    "upset and felt that her privacy had been violated. She decided to seek legal "
    "advice regarding the breach of the agreed-upon restriction on her PHI. \n"
    " ### Response: HIPAA Privacy Rule forbids the case. \n"
    " ### Input: "
)

APPLICABILITY_FEW_SHOT = (
    "Read the case: Jane is a factory worker at XYZ Manufacturing, a company that "
    "requires its employees to undergo annual medical check-ups due to the nature "
    "of their work. The company asks Dr. Smith, a covered health care provider, to "
    "perform these check-ups, including medical surveillance of the workplace to "
    "ensure a safe environment. During Jane's check-up, Dr. Smith discovers that "
    "Jane has a medical condition that could be related to her workplace "
    "environment. Dr. Smith believes that it is important to inform XYZ "
    "Manufacturing about Jane's condition to address any potential workplace "
    "hazards. Dr. Smith contacts the HR department of XYZ Manufacturing and shares "
    "Jane's medical information related to the discovered condition. The HR "
    "department acknowledges the information and initiates an internal "
    "investigation to ensure the safety of their employees. Jane is informed about "
    "the disclosure of her medical information to her employer by Dr. Smith and "
    "agrees to the disclosure for the purpose of workplace safety evaluation.\n"
    " ### Response: HIPAA Privacy Rule is applicable to the case. \n"
    " ### Input: Read the case: A state trooper was approached during a call by a "
    "bystander, who informed the officer that her estranged spouse had a "
    "significant amount of marijuana at his residence. A local sheriff, also "
    "present at the scene, corroborated that he had heard similar rumors about the "
    "spouse and had previously received information from the bystander when working "
    "undercover. Acting on this information, the trooper prepared an affidavit and "
    "obtained a search warrant from a trial commissioner. During the execution of "
    "the warrant, the officers discovered marijuana and drug paraphernalia in the "
    "spouse's home. The spouse contested the legality of the search, arguing that "
    "the affidavit was insufficient for the issuance of the warrant due to a lack "
    "of specific details about the contraband, inaccuracies in the property "
    "owner's name, and insufficient evidence of the informant's reliability. "
    "However, the trial court determined that the officer's testimony met the "
    "standards required for a good faith exception to the warrant requirement, and "
    "thus denied the motion to suppress the evidence, which is the subject of the "
    "current appeal. \n"
    " ### Response: HIPAA Privacy Rule is not applicable to the case. \n"
    " ### Input: "
)


# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------

def build_prompt_for_row(
    row: Dict[str, Any],
    task: str,
    mode: str,
    few_shot: bool = False,
) -> str:
    """Assemble the full Alpaca-format prompt for a single case.

    Args:
        row: Dict with at least ``generate_background`` key.
        task: "compliance" or "applicability".
        mode: "direct" or "cot".
        few_shot: Whether to prepend few-shot examples (direct mode only).

    Returns:
        Complete prompt string ready for LLM inference.
    """
    case_text = str(row.get("generate_background", ""))

    # Select instruction
    if task == "compliance":
        if mode == "cot":
            instruction = COMPLIANCE_COT_INSTRUCTION
        else:
            instruction = COMPLIANCE_DIRECT_INSTRUCTION
    elif task == "applicability":
        if mode == "cot":
            instruction = APPLICABILITY_COT_INSTRUCTION
        else:
            instruction = APPLICABILITY_DIRECT_INSTRUCTION
    else:
        raise ValueError(f"Unknown task: {task!r}. Expected 'compliance' or 'applicability'.")

    # Build input text
    input_text = f"Read the case: {case_text}"

    # Prepend few-shot examples (direct mode only)
    if few_shot and mode == "direct":
        if task == "compliance":
            input_text = COMPLIANCE_FEW_SHOT + input_text
        else:
            input_text = APPLICABILITY_FEW_SHOT + input_text

    # Format with Alpaca template
    return ALPACA_TEMPLATE_WITH_INPUT.format(
        instruction=instruction,
        input=input_text,
    )
