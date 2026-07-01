import json
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.services.agent.tools import TOOL_SCHEMAS, execute_tool
from app.services.agent.guardrails import check_step, GuardrailViolation

_openai = None


def _llm():
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=get_settings().openai_api_key)
    return _openai


SYSTEM_PROMPT = """You are ResearchMind, an expert research assistant.
You have access to a knowledge base of documents and tools to search it.
1. THINK about what information you need
2. CALL one tool at a time
3. OBSERVE the result
4. Repeat until you can answer fully
5. Call finish() with your complete answer
Rules:
- Call ONE tool at a time, then wait for the result
- Ground your answer in retrieved sources
- If a search returns nothing, try different terms
- Do NOT make up information
"""


async def run_react_agent(task, max_steps=None):
    settings = get_settings()
    max_steps = max_steps or settings.agent_max_steps

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
    ]

    steps = []
    tool_call_history = []
    final_answer = ""

    for step_num in range(1, max_steps + 1):
        response = await _llm().chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.1,
        )

        message = response.choices[0].message

        assistant_msg = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            final_answer = message.content or ""
            break

        should_finish = False
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_input = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            try:
                check_step(
                    step_num=step_num,
                    thought=message.content or "",
                    tool_name=tool_name,
                    tool_input=tool_input,
                    max_steps=max_steps,
                    tool_call_history=tool_call_history,
                )
            except GuardrailViolation as e:
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"Guardrail stopped: {e.reason}"})
                final_answer = f"Agent stopped: {e.reason}"
                should_finish = True
                continue

            if tool_name == "finish":
                final_answer = tool_input.get("answer", "")
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": "Task completed."})
                should_finish = True
                continue

            observation = await execute_tool(tool_name, tool_input)
            tool_call_history.append((tool_name, str(sorted(tool_input.items()))))

            steps.append({
                "step": step_num,
                "thought": message.content or f"Calling {tool_name}",
                "action": tool_name,
                "action_input": tool_input,
                "observation": observation[:1500],
            })

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": observation})

        if should_finish:
            break

    if not final_answer and steps:
        final_answer = "Agent reached max steps. Last observation:\n" + steps[-1]["observation"]

    return {"task": task, "answer": final_answer, "steps": steps, "total_steps": len(steps), "sources": []}
