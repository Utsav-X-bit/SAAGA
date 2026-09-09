"""
SAAGA Engine Prompting & Text Utilities
=======================================
Chat templating, think-block stripping, system prompt truncation, and token management.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

# Thinking-model tag patterns
_OPEN_TAGS = ("<think>",)
_CLOSE_TAGS = ("</think>",)
_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_think_blocks(text: str | None) -> str | None:
    """Strip <think>...</think> reasoning blocks from thinking models (Qwen, DeepSeek-R1)."""
    if text is None:
        return None
    if not text:
        return ""

    out = text
    for open_tag in _OPEN_TAGS:
        while True:
            low = out.lower()
            start = low.find(open_tag)
            if start == -1:
                break
            match = _CLOSE_RE.search(out, pos=start)
            if match:
                end = match.end()
                out = out[:start] + out[end:]
            else:
                out = out[:start]
                break
    return out.strip()


def strip_few_shot_patterns(text: str) -> str:
    """Clean few-shot artifacts, delimiters, and think blocks from model responses."""
    if not text:
        return ""
    text = strip_think_blocks(text)
    # Remove few-shot delimiter emojis and artifacts
    cleaned = re.sub(r"~\U0001F4A9~.*?~\U0001FAF7~", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"Response:\s*I love Taylor!", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"USER:\s*I love Taylor!", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def truncate_system_content_to_fit(
    messages: Sequence[dict[str, str]],
    tokenizer: Any,
    max_prompt_tokens: int = 3072,
) -> list[dict[str, str]]:
    """Truncate system prompt content if the total conversation exceeds token budget."""
    if not messages:
        return []

    out_messages = [dict(m) for m in messages]
    if tokenizer is None:
        return out_messages

    try:
        raw_prompt = tokenizer.apply_chat_template(out_messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(raw_prompt, add_special_tokens=False)
        total_len = len(tokens)
    except Exception:
        return out_messages

    if total_len <= max_prompt_tokens:
        return out_messages

    overflow = total_len - max_prompt_tokens
    for msg in out_messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if len(content) > overflow * 4:
                # Trim from the middle of system defense
                half = (len(content) - (overflow * 4)) // 2
                msg["content"] = content[:half] + "\n...[truncated]...\n" + content[-half:]
            break

    return out_messages


def apply_chat_template_safe(
    messages: Sequence[dict[str, str]],
    tokenizer: Any,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
) -> str | list[int]:
    """Safely apply chat template with fallback to simple string concatenation."""
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass

    # Fallback template
    formatted = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        formatted.append(f"{role}: {content}")
    if add_generation_prompt:
        formatted.append("Assistant:")
    return "\n\n".join(formatted)
