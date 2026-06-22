# Vendored from CI-RL (github.com/EricGLan/CI-RL) - Apache 2.0 License
from procoder.functional import format_multiple_prompts, add_refnames
from .shared import *

AGENT_PRIVACY_ENHANCED_PROMPT = (
    Collection(
        AGENT_ENV_SETUP, AGENT_TASK_DESC, AGENT_FORMAT_INSTRUCTION, AGENT_TASK_BEGIN
    )
    .set_indexing_method(sharp2_indexing)
    .set_sep("\n\n")
)

AGENT_PRIVACY_ENHANCED_PROMPT = replace_agent_ref_with_pronoun(AGENT_PRIVACY_ENHANCED_PROMPT)
