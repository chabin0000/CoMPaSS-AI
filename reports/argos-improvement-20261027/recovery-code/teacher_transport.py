import json,re,time,urllib.request
class OpenAICompatibleTeacher:
 def __init__(self,config):
  self.config=config;self.base_url=config["base_url"].rstrip("/");self.timeout=float(config["timeout_seconds"])

def extract_final(message, finish_reason="stop"):
    if finish_reason not in {"stop", "eos"}:
        raise ValueError(f"unfinished/refused generation: {finish_reason}")
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(x.get("text", "") for x in content if x.get("type") == "text")
    if not isinstance(content, str):
        raise ValueError("invalid final content")
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    # Some engines return Harmony verbatim; require the final channel explicitly.
    if "<|channel|>" in content or "<|start|>" in content:
        match = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|fim_suffix\|>|<\|end\|>|$)", content, re.S)
        if not match:
            raise ValueError("Harmony final channel missing")
        reasoning = str(reasoning) + content[:match.start()]
        content = match.group(1)
    if content.count("<think>") != content.count("</think>"):
        raise ValueError("incomplete thinking output")
    thoughts = re.findall(r"<think>(.*?)</think>", content, re.S)
    reasoning = str(reasoning) + "\n".join(thoughts)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    if not content or "<|" in content or "</think>" in content:
        raise ValueError("empty or unparsed final output")
    return content, reasoning


class TeacherGenerationError(ValueError):
    """Invalid/incomplete output with diagnostics, never eligible as an answer."""
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


class StreamingTeacher(OpenAICompatibleTeacher):
    """Reuse transport conventions; request stream to measure TTFT, retain full raw events privately."""
    def generate_record(self, question):
        payload = {"model": self.config["model"], "messages": [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": question["question"]}],
            **self.config["generation"], **self.config.get("extra_body", {}),
            "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        first, final_first, content, reasoning, usage, events, finish = None, None, "", "", {}, [], None
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            for line in response:
                if time.perf_counter() - start > self.timeout:
                    raise TimeoutError("total generation deadline exceeded")
                if not line.startswith(b"data:"):
                    continue
                text = line[5:].strip()
                if text == b"[DONE]":
                    break
                event = json.loads(text)
                events.append(event)
                if event.get("error"):
                    raise ValueError("server returned generation error")
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    token_text = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if (token_text or thought) and first is None:
                        first = time.perf_counter() - start
                    if token_text and final_first is None:
                        final_first = time.perf_counter() - start
                    content += token_text
                    reasoning += thought
                    finish = choice.get("finish_reason") or finish
        try:
            final, thinking = extract_final({"content": content, "reasoning_content": reasoning}, finish)
        except ValueError as error:
            raise TeacherGenerationError(str(error), {
                "response_valid": False, "eligible_for_research_training": False,
                "partial_final_content": content, "partial_reasoning": reasoning,
                "raw_events": events, "finish_reason": finish, "generation": payload,
                "latency_seconds": time.perf_counter() - start, "ttft_seconds": first,
                "final_ttft_seconds": final_first, "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"), "model_metadata": self.config["model_metadata"],
                "created_at": time.time(), "mode": self.config.get("mode", "actual")}) from error
        elapsed = time.perf_counter() - start
        return {"final_answer": final, "reasoning": thinking, "raw_events": events, "finish_reason": finish,
                "latency_seconds": elapsed, "ttft_seconds": first,
                "final_ttft_seconds": final_first, "ttft_kind": "first_reasoning_or_final_token",
                "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                "generation": payload, "model_metadata": self.config["model_metadata"],
                "created_at": time.time(), "mode": self.config.get("mode", "actual")}

