import llm
import httpx
import json
from typing import Optional, List, Union, Dict
from pydantic import Field

DEFAULT_ALIASES = {
    "gemma2-9b-it": "groq-gemma2-9b",
    "gemma-7b-it": "groq-gemma-7b",
    "llama-3.1-70b-versatile": "groq-llama3.1-70b",
    "llama-3.1-8b-instant": "groq-llama3.1-8b",
    "llama3-70b-8192": "groq-llama3-70b",
    "llama3-8b-8192": "groq-llama3-8b",
    "llama3-groq-70b-8192-tool-use-preview": "groq-llama3-70b-tool",
    "llama3-groq-8b-8192-tool-use-preview": "groq-llama3-8b-tool",
    "llama-guard-3-8b": "groq-llama-guard",
    "mixtral-8x7b-32768": "groq-mixtral"
}

def fetch_groq_models():
    key = llm.get_key("", "groq", "LLM_GROQ_KEY")
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {key}"}
    response = httpx.get(url, headers=headers)
    response.raise_for_status()
    return response.json()["data"]

@llm.hookimpl
def register_models(register):
    for model in fetch_groq_models():
        model_id = model["id"]
        alias = DEFAULT_ALIASES.get(model_id)
        aliases = [alias] if alias else []
        if model_id == "whisper-large-v3":
            register(GroqWhisper(model_id), aliases=aliases)
        else:
            register(Groq(model_id), aliases=aliases)

class Groq(llm.Model):
    can_stream = True

    class Options(llm.Options):
        temperature: Optional[float] = Field(default=1)
        top_p: Optional[float] = Field(default=1)
        max_tokens: Optional[int] = Field(default=None)
        stop: Optional[Union[str, List[str]]] = Field(default=None)
        frequency_penalty: Optional[float] = Field(default=0)
        presence_penalty: Optional[float] = Field(default=0)
        response_format: Optional[str] = Field(default=None)
        tools: Optional[List[Dict[str, Union[str, Dict[str, str]]]]] = Field(default=None)
        tool_choice: Optional[Union[str, Dict[str, str]]] = Field(default=None)

    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation):
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        if conversation:
            for response in conversation.responses:
                messages.append({"role": "user", "content": response.prompt.prompt})
                messages.append({"role": "assistant", "content": response.text()})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "groq", "LLM_GROQ_KEY")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

        messages = self.build_messages(prompt, conversation)

        body = {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
            **{k: v for k, v in prompt.options.dict().items() if v is not None}
        }

        if prompt.options.response_format == "json":
            body["response_format"] = {"type": "json_object"}
            body["stream"] = False

        with httpx.stream("POST", url, json=body, headers=headers) as r:
            if stream and not body.get("response_format"):
                for chunk in r.iter_lines():
                    if chunk.startswith(b"data: "):
                        chunk = json.loads(chunk[6:])
                        if chunk["choices"][0]["delta"].get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
            else:
                response_json = r.json()
                content = response_json["choices"][0]["message"]["content"]
                if prompt.options.response_format == "json":
                    content = json.dumps(json.loads(content), indent=2)
                yield content

class GroqWhisper(llm.Model):
    can_stream = False

    class Options(llm.Options):
        prompt: Optional[str] = Field(default=None)
        response_format: Optional[str] = Field(default="json")
        temperature: Optional[float] = Field(default=0, ge=0, le=1)
        language: Optional[str] = Field(default=None)

    def __init__(self, model_id):
        self.model_id = model_id

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "groq", "LLM_GROQ_KEY")
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {key}"}
        data = {
            "model": self.model_id,
            **{k: v for k, v in prompt.options.dict().items() if v is not None}
        }
        response = httpx.post(url, headers=headers, data=data)
        response.raise_for_status()
        yield response.json()["text"]
