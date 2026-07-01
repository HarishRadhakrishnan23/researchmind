"""
Agent guardrails — safety and cost controls.
Called at each ReAct step before executing a tool.
"""

import re
from dataclasses import dataclass


DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bdrop\s+table\b",
    r"\bexec\s*\(",
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"<script",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]


@dataclass
class GuardrailViolation(Exception):
    reason: str


def check_step(
    step_num: int,
    thought: str,
    tool_name: str,
    tool_input: dict,
    max_steps: int,
    tool_call_history: list[tuple[str, str]],
) -> None:
    """
    Raises GuardrailViolation if any check fails.
    Call this BEFORE executing each tool.
    """
    # 1. Max step limit
    if step_num > max_steps:
        raise GuardrailViolation(
            f"Agent exceeded maximum steps ({max_steps}). Stopping."
        )

    # 2. Dangerous content in thought or tool input
    content_to_check = thought + str(tool_input)
    for pattern in _COMPILED:
        if pattern.search(content_to_check):
            raise GuardrailViolation(
                f"Dangerous pattern detected in agent output: {pattern.pattern}"
            )

    # 3. Infinite loop detection — same tool + input repeated 3x
    key = (tool_name, str(sorted(tool_input.items())))
    repeat_count = sum(1 for h in tool_call_history if h == key)
    if repeat_count >= 2:
        raise GuardrailViolation(
            f"Loop detected: tool '{tool_name}' called with identical inputs 3 times."
        )
