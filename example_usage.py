#!/usr/bin/env python3
"""
Example usage of the Deep Research Agent (deepagents-based)
"""

import asyncio
import os

from deep_research_graph import deep_research_graph_for_api


async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set GOOGLE_API_KEY: export GOOGLE_API_KEY='your-key'")
        return

    query = "What are the latest developments in quantum computing and their potential impact on cryptography?"
    print(f"Starting research: {query}\n")

    result = await deep_research_graph_for_api.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": "example-session"}},
    )

    print(result["messages"][-1].content)


async def troubleshoot_trace_truncation():
    """Reproduce and fix trace truncation caused by a custom message reducer.

    A customer reported that their LangSmith traces were being truncated. They
    were using a custom message reducer (add_messages_with_timestamp) that only
    added created_at timestamps to additional_kwargs, but passed msg.content
    through untouched.

    The main issue: when the LLM returns structured content (a list of content
    block dicts, e.g. [{"type": "text", ...}, {"type": "tool_use", ...}]), the
    original reducer left that list as-is in message history. The
    SummarizationMiddleware serializes each message — including its content
    field — to estimate token counts. A structured content list serializes to
    far more tokens than the equivalent plain text string. Over many turns this
    bloat compounds, the middleware sees an inflated token count, triggers
    summarization early, and older messages (and their trace spans) get
    collapsed.

    The timestamps in additional_kwargs were a minor contributing factor (~50
    chars per message), but the real culprit was the un-normalized structured
    content accumulating across turns.
    """
    from datetime import UTC, datetime

    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
    from langgraph.graph.message import add_messages

    # ------------------------------------------------------------------
    # ORIGINAL reducer (customer's code) — only adds timestamps,
    # passes structured content through untouched
    # ------------------------------------------------------------------
    def add_messages_with_timestamp_original(left, right):
        right_list = right if isinstance(right, list) else [right]
        processed_right = []
        for msg in right_list:
            if isinstance(msg, AIMessage) and "created_at" not in msg.additional_kwargs:
                updated_kwargs = {
                    **msg.additional_kwargs,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                processed_msg = msg.model_copy(update={"additional_kwargs": updated_kwargs})
                processed_right.append(processed_msg)
            else:
                processed_right.append(msg)
        return add_messages(left, processed_right)

    # ------------------------------------------------------------------
    # FIXED reducer — normalizes structured content on completed
    # AIMessages so the middleware sees realistic token counts
    # ------------------------------------------------------------------
    def _process_ai_message(msg):
        content = msg.content
        # For completed AIMessages (not chunks), if content is structured
        # (list of content block dicts), collapse it to plain text via
        # msg.text. This prevents bloated serialization in the middleware.
        # Chunks are left untouched so streaming merge logic still works.
        if not isinstance(msg, AIMessageChunk) and not isinstance(content, str):
            content = msg.text
        kwargs = dict(msg.additional_kwargs)
        if "created_at" not in kwargs:
            kwargs["created_at"] = datetime.now(UTC).isoformat()
        return msg.model_copy(update={"content": content, "additional_kwargs": kwargs})

    def add_messages_with_timestamp_fixed(left, right):
        right_list = right if isinstance(right, list) else [right]
        processed = [
            _process_ai_message(msg) if isinstance(msg, AIMessage) else msg
            for msg in right_list
        ]
        return add_messages(left, processed)

    # ------------------------------------------------------------------
    # Simulate a multi-turn conversation with structured content
    # ------------------------------------------------------------------
    from langchain_core.messages import get_buffer_string

    def build_history(reducer, label):
        """Build 20-turn history using the given reducer and measure size."""
        messages = []
        for i in range(20):
            messages = reducer(
                messages,
                [HumanMessage(content=f"Question {i}: Tell me about topic {i}.")],
            )
            # Simulate an AIMessage with structured content (list of blocks),
            # like what the LLM actually returns when using tool calls
            structured_content = [
                {"type": "text", "text": f"Here is a detailed answer about topic {i}. " * 20},
                {"type": "tool_use", "id": f"call_{i}", "name": "write_file",
                 "input": {"path": f"/findings/step_{i}.md", "content": f"Finding {i} " * 30}},
            ]
            messages = reducer(
                messages,
                [AIMessage(content=structured_content)],
            )

        # Measure what the SummarizationMiddleware actually sees:
        # it serializes the full message dicts (including content field)
        # to estimate token counts — not just the text portion.
        import json
        serialized_size = sum(
            len(json.dumps(m.content, default=str)) + len(json.dumps(m.additional_kwargs, default=str))
            for m in messages
        )
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        structured_count = sum(
            1 for m in messages
            if isinstance(m, AIMessage) and isinstance(m.content, list)
        )
        return {
            "label": label,
            "total_messages": len(messages),
            "ai_messages": ai_count,
            "still_structured": structured_count,
            "serialized_chars": serialized_size,
            "approx_tokens": serialized_size // 4,
        }

    original = build_history(add_messages_with_timestamp_original, "ORIGINAL (bug)")
    fixed = build_history(add_messages_with_timestamp_fixed, "FIXED")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=== Trace Truncation Troubleshooting ===\n")

    print("Problem: Customer's LangSmith traces were being truncated.\n")
    print("Root cause: The original reducer passed msg.content through untouched.")
    print("When the LLM returns structured content (a list of content block dicts")
    print("like [{\"type\": \"text\", ...}, {\"type\": \"tool_use\", ...}]), that list")
    print("serializes to far more tokens than the equivalent plain text. Over many")
    print("turns, this bloat compounds — the SummarizationMiddleware sees an inflated")
    print("token count, triggers early, and older messages/trace spans get collapsed.\n")

    for stats in [original, fixed]:
        print(f"--- {stats['label']} ---")
        print(f"  Total messages:              {stats['total_messages']}")
        print(f"  AI messages:                 {stats['ai_messages']}")
        print(f"  Still have structured content: {stats['still_structured']}")
        print(f"  Serialized size (what MW sees): {stats['serialized_chars']} chars (~{stats['approx_tokens']} tokens)")
        print()

    reduction = original["serialized_chars"] - fixed["serialized_chars"]
    pct = (reduction / original["serialized_chars"]) * 100 if original["serialized_chars"] else 0
    print(f"Size reduction:  {reduction} chars ({pct:.0f}% smaller)\n")

    print("Fix: The refactored reducer calls msg.text on completed AIMessages (not")
    print("chunks) to collapse structured content blocks into a plain text string")
    print("before they accumulate in history. Chunks are left untouched so streaming")
    print("merge logic still works. This gives the SummarizationMiddleware a realistic")
    print("token count, preventing premature summarization and trace truncation.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--troubleshoot":
        asyncio.run(troubleshoot_trace_truncation())
    else:
        asyncio.run(main())
