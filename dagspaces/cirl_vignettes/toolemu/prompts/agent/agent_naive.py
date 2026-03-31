# Vendored from CI-RL (github.com/EricGLan/CI-RL) - Apache 2.0 License
from procoder.functional import format_multiple_prompts, add_refnames
from procoder.functional import replaced_submodule
from .shared import *

AGENT_NAIVE_PROMPT = (
    Collection(
        AGENT_ENV_SETUP, AGENT_TASK_DESC, AGENT_FORMAT_INSTRUCTION, AGENT_TASK_BEGIN
    )
    .set_indexing_method(sharp2_indexing)
    .set_sep("\n\n")
)

AGENT_NAIVE_PROMPT = replace_agent_ref_with_pronoun(AGENT_NAIVE_PROMPT)
AGENT_NAIVE_CLAUDE_PROMPT = replaced_submodule(
    AGENT_NAIVE_PROMPT, "task_begin", AGENT_TASK_BEGIN_FOR_CLAUDE
)
