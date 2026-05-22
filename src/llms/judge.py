import json
from typing import Any, Iterable, List, Dict, Optional, Tuple


def _extract_json_by_braces(
    text: str,
    required_keys: Optional[Iterable[str]] = None,
) -> Tuple[Optional[dict], int]:
    """Extract a JSON object from text by brace matching.

    Returns (parsed_dict, start_idx) on success, (None, -1) on failure.
    Tries candidates from innermost to outermost; first valid parse is returned.
    """
    brace_count = 0
    start_idx = -1
    candidates = []

    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                candidates.append((start_idx, i + 1, text[start_idx : i + 1]))
                start_idx = -1

    for candidate_start_idx, _end_idx, candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                continue
            if required_keys is not None and not any(k in parsed for k in required_keys):
                continue
            return parsed, candidate_start_idx
        except json.JSONDecodeError:
            continue

    return None, -1


def extract_json_from_response(
    text: str,
    required_keys: Optional[Iterable[str]] = None,
) -> Tuple[Any, int]:
    """Extract JSON from an LLM response wrapped in markdown fences or raw text.

    Tries ```json fenced blocks, generic ``` blocks, then brace-matched objects.

    Args:
        text: Raw LLM response text.
        required_keys: Optional keys that must be present in a parsed dict.
            When provided, at least one key must match.

    Returns:
        Tuple of (parsed_json, start_index) where start_index is the character
        position where the JSON block begins in the stripped text.

    Raises:
        ValueError: If text is empty or no valid JSON is found.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Empty or invalid response text")

    s = text.strip()
    if not s:
        raise ValueError("Empty response text")

    def _valid_dict(parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            return False
        if required_keys is None:
            return True
        return any(k in parsed for k in required_keys)

    idx = s.find("```json")
    if idx >= 0:
        start = idx + len("```json")
        end = s.find("```", start)
        if end == -1:
            end = len(s)
        block = s[start:end].strip()
        try:
            parsed = json.loads(block)
            if _valid_dict(parsed):
                return parsed, idx
        except json.JSONDecodeError:
            pass

    idx = s.find("```")
    while idx >= 0:
        start = idx + 3
        end = s.find("```", start)
        if end == -1:
            end = len(s)
        block = s[start:end].strip()
        if block.startswith("json"):
            block = block[4:].lstrip()
        if block.startswith("{"):
            try:
                parsed = json.loads(block)
                if _valid_dict(parsed):
                    return parsed, idx
            except json.JSONDecodeError:
                pass
        idx = s.find("```", end + 3) if end < len(s) else -1

    parsed, json_start = _extract_json_by_braces(s, required_keys)
    if parsed is not None:
        return parsed, json_start

    raise ValueError("No valid JSON found in response")


def parse_llm_json_response(
    text: str,
    required_keys: Optional[Iterable[str]] = None,
) -> Any:
    """Parse JSON from an LLM response, ignoring surrounding markdown."""
    parsed, _ = extract_json_from_response(text, required_keys)
    return parsed


class GenericJudgeLLM:
    def __init__(self, any_llm):
        self.llm = any_llm

    def generate(self, prompt: str):
        raw = self.llm.generate(prompt)
        return raw

    async def a_generate(self, prompt: str):
        return await self.llm.a_generate(prompt)


class ChatJudgeLLM:
    def __init__(self, any_llm, system_prompt: str = None):
        self.llm = any_llm
        self.conversation_history: List[Dict[str, str]] = []
        
        # Set up system prompt if provided
        if system_prompt:
            self.conversation_history.append({
                "role": "system", 
                "content": system_prompt
            })
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate response using chat completion
        response = self.llm.generate_chat_completion(
            messages=self.conversation_history,
            **kwargs
        )
        
        # Add assistant response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response

    async def a_generate(self, prompt: str, **kwargs) -> str:
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate response using async chat completion
        response = await self.llm.a_generate_chat_completion(
            messages=self.conversation_history,
            **kwargs
        )
        
        # Add assistant response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        system_prompt = None
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            system_prompt = self.conversation_history[0]["content"]
        
        self.conversation_history = []
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def add_system_message(self, message: str):
        # Remove existing system message if any
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history.pop(0)
        
        # Add new system message at the beginning
        self.conversation_history.insert(0, {
            "role": "system",
            "content": message
        })
    
    def get_conversation_summary(self) -> str:
        summary = f"Conversation has {len(self.conversation_history)} messages:\n"
        for i, msg in enumerate(self.conversation_history):
            role = msg["role"]
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            summary += f"{i+1}. {role}: {content}\n"
        return summary
