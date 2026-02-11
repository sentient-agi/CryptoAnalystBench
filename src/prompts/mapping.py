from src.prompts import citation_check
from src.prompts import hallucination_check

EVAL_PROMPT_MAPPING = {
    "citation_check": {
        "system_prompt":citation_check.SYSTEM_PROMPT,
        "user_prompt":citation_check.USER_PROMPT,
        "variables":["<response>","<context>"]
        },
    "hallucination_check": {
        "system_prompt":hallucination_check.SYSTEM_PROMPT,
        "user_prompt":hallucination_check.USER_PROMPT,
        "variables":["<response>","<context>"]
        },
}