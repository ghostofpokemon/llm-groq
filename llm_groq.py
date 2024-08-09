import llm
from groq import Groq
from pydantic import Field
from typing import Optional, List, Union, Dict
import httpx
import os
import json

DEFAULT_ALIASES = {
    "gemma-7b-it": "groq-gemma",
    "gemma2-9b-it": "groq-gemma2",
    "llama2-70b-4096": "groq-llama2",
    "llama3-8b-8192": "groq-llama3",
    "llama3-70b-8192": "groq-llama3-70b",
    "mixtral-8x7b-32768": "groq-mixtral",
    "llama-3.1-8b-instant": "groq-llama3.1-8b",
    "llama-3.1-70b-versatile": "groq-llama3.1-70b",
    "llama-3.1-405b-reasoning": "groq-llama3.1-405b",
}

def fetch_models():
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    response = httpx.get(url, headers=headers)
    response.raise_for_status()
    return response.json()["data"]

@llm.hookimpl
def register_models(register):
    for model_id in get_model_ids():
        alias = DEFAULT_ALIASES.get(model_id)
        aliases = [alias] if alias else []
        if model_id == "whisper-large-v3":
            register(LLMGroqWhisper(model_id), aliases=aliases)
        else:
            register(LLMGroq(model_id), aliases=aliases)

def get_model_ids():
    models = fetch_models()
    return [model["id"] for model in models]

class LLMGroq(llm.Model):
    can_stream = True

    class Options(llm.Options):
        temperature: Optional[float] = Field(default=1)
        top_p: Optional[float] = Field(default=1)
        max_tokens: Optional[int] = Field(default=None)
        stop: Optional[Union[str, List[str]]] = Field(default=None)
        frequency_penalty: Optional[float] = Field(default=0)
        presence_penalty: Optional[float] = Field(default=0)
        user: Optional[str] = Field(default=None)
        seed: Optional[int] = Field(default=None)
        response_format: Optional[str] = Field(default=None)
        stream: Optional[bool] = Field(default=True)
        tool_choice: Optional[Union[str, Dict[str, str]]] = Field(default=None)
        tools: Optional[List[Dict[str, Union[str, Dict[str, str]]]]] = Field(default=None)
        parallel_tool_calls: Optional[bool] = Field(default=True)

    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation):
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            return messages
        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "groq", "LLM_GROQ_KEY")
        messages = self.build_messages(prompt, conversation)
        client = Groq(api_key=key)

        # Ensure the prompt includes instructions for generating JSON
        if prompt.options.response_format == "json":
            json_instruction = "Please provide the response in JSON format."
            if not any(msg["content"] == json_instruction for msg in messages):
                messages.append({"role": "user", "content": json_instruction})

        # Convert response_format string to dictionary if necessary
        response_format_dict = None
        if prompt.options.response_format:
            if prompt.options.response_format == "json":
                response_format_dict = {"type": "json_object"}
            elif prompt.options.response_format == "verbose_json":
                response_format_dict = {"type": "verbose_json"}
            else:
                raise ValueError("Invalid response_format value")

        # Prepare the request body
        body = {
            "messages": messages,
            "model": self.model_id,
            "stream": stream if not response_format_dict else False,
            "temperature": prompt.options.temperature,
            "top_p": prompt.options.top_p,
            "max_tokens": prompt.options.max_tokens,
            "stop": prompt.options.stop,
            "frequency_penalty": prompt.options.frequency_penalty,
            "presence_penalty": prompt.options.presence_penalty,
            "user": prompt.options.user,
            "seed": prompt.options.seed,
            "response_format": response_format_dict,
            "tools": prompt.options.tools,
            "parallel_tool_calls": prompt.options.parallel_tool_calls,
        }

        # Only include tool_choice if it has a valid value
        if prompt.options.tool_choice is not None:
            body["tool_choice"] = prompt.options.tool_choice

        resp = client.chat.completions.create(**body)

        if stream and not response_format_dict:
            for chunk in resp:
                if chunk.choices[0].delta.content:
                    yield from chunk.choices[0].delta.content
        else:
            # Handle JSON response
            json_response = json.loads(resp.choices[0].message.content)
            yield json.dumps(json_response, indent=2)  # Convert JSON to string with indentation

class LLMGroqWhisper(llm.Model):
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
        client = Groq(api_key=key)
        resp = client.transcribe.create(
            model=self.model_id,
            prompt=prompt.options.prompt,
            response_format=prompt.options.response_format,
            temperature=prompt.options.temperature,
            language=prompt.options.language,
        )
        yield resp.text
