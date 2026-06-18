import json
from pathlib import Path
from typing import Callable

from langchain_core.prompts import PromptTemplate, load_prompt
from pydantic import ValidationError

TEMPLATE_PATH = Path(__file__).resolve().parent / "template.json"

FALLBACK_PROMPT = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template=(
        'You are a research assistant. Summarize the paper "{paper_input}" '
        "in a {style_input} style. Target length: {length_input}.\n\n"
        "Highlight the main problem, approach, and contributions."
    ),
)


def load_research_prompt(
    on_warning: Callable[[str], None] | None = None,
) -> PromptTemplate:
    def warn(message: str) -> None:
        if on_warning is not None:
            on_warning(message)

    try:
        loaded = load_prompt(str(TEMPLATE_PATH))
        if not isinstance(loaded, PromptTemplate):
            warn("Loaded prompt is not a simple `PromptTemplate`; using built-in default.")
            return FALLBACK_PROMPT
        return loaded
    except FileNotFoundError:
        warn(f"Prompt file not found at `{TEMPLATE_PATH}`. Using built-in default template.")
        return FALLBACK_PROMPT
    except json.JSONDecodeError as exc:
        warn(f"`template.json` is not valid JSON ({exc}). Using built-in default template.")
        return FALLBACK_PROMPT
    except (ValueError, OSError, ValidationError) as exc:
        warn(f"Could not load prompt from `template.json` ({exc}). Using built-in default template.")
        return FALLBACK_PROMPT
