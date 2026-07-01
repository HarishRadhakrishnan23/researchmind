"""
Tool definitions for the ReAct agent.
Each tool has:
  - A JSON schema (sent to LLM for function calling)
  - An async executor function
 
To add a new tool:
  1. Add its schema to TOOL_SCHEMAS
  2. Add its async handler in execute_tool()
"""
 
from typing import Any
from app.services.rag.pipeline import query_rag
 
# ── Tool schemas ──────────────────────────────────────────────────────────────
 
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the document knowledge base for specific information. "
                "Use targeted, specific queries for best results. "
                "Call multiple times with different queries to gather different facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query — be precise, not vague",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (1-10, default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": (
                "Get a synthesized summary of everything in the knowledge base about a topic. "
                "Better for broad overviews. Use rag_search for specific facts or quotes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to summarize from documents",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_and_analyze",
            "description": (
                "Search for two different aspects and return both for comparison. "
                "Use when you need to compare, contrast, or relate two concepts from documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "aspect_a": {
                        "type": "string",
                        "description": "First aspect to search for",
                    },
                    "aspect_b": {
                        "type": "string",
                        "description": "Second aspect to search for",
                    },
                },
                "required": ["aspect_a", "aspect_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this ONLY when you have enough information to fully answer the task. "
                "Provide a complete, well-structured answer with source references."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete final answer to the user's task",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]
 
 
# ── Tool executors ────────────────────────────────────────────────────────────
 
async def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """
    Dispatch tool call to the correct async handler.
    Returns a string observation to feed back into the LLM context.
    """
 
    if tool_name == "rag_search":
        result = await query_rag(
            query=tool_input["query"],
            top_k=tool_input.get("top_k", 5),
        )
        if not result["sources"]:
            return f"No results found for query: '{tool_input['query']}'. Try different keywords."
 
        chunks_text = "\n\n".join(
            f"[{s['filename']}, page {s['page']}]:\n{s['text']}"
            for s in result["sources"]
        )
        return f"Found {len(result['sources'])} relevant chunks:\n\n{chunks_text}"
 
    elif tool_name == "summarize_topic":
        result = await query_rag(
            query=f"Summarize everything about: {tool_input['topic']}",
            top_k=8,
        )
        answer = result.get("answer", "")
        if not answer or "No relevant" in answer:
            return f"No information found about '{tool_input['topic']}' in the knowledge base."
        return f"Summary of '{tool_input['topic']}':\n\n{answer}"
 
    elif tool_name == "compare_and_analyze":
        result_a = await query_rag(query=tool_input["aspect_a"], top_k=4)
        result_b = await query_rag(query=tool_input["aspect_b"], top_k=4)
 
        text_a = "\n".join(s["text"][:300] for s in result_a["sources"]) or "No results"
        text_b = "\n".join(s["text"][:300] for s in result_b["sources"]) or "No results"
 
        return (
            f"=== Aspect A: {tool_input['aspect_a']} ===\n{text_a}\n\n"
            f"=== Aspect B: {tool_input['aspect_b']} ===\n{text_b}"
        )
 
    elif tool_name == "finish":
        return tool_input.get("answer", "")
 
    else:
        return f"Unknown tool '{tool_name}'. Available: rag_search, summarize_topic, compare_and_analyze, finish"
 