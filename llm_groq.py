import llm
import httpx
import json
import os
import time
from typing import Optional, List, Union, Dict
from pydantic import Field
from pathlib import Path

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

def fetch_cached_json(url, path, cache_timeout, headers):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        mod_time = path.stat().st_mtime
        if time.time() - mod_time < cache_timeout:
            with open(path, "r") as file:
                return json.load(file)
    try:
        response = httpx.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
        with open(path, "w") as file:
            json.dump(data, file)
        return data
    except httpx.HTTPError as e:
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            raise Exception(f"Failed to download data: {str(e)}")

def fetch_groq_models():
    key = llm.get_key("", "groq", "LLM_GROQ_KEY")
    if not key:
        raise Exception("Groq API key not found. Please set the LLM_GROQ_KEY environment variable.")
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {key}"}
    cache_path = llm.user_dir() / "groq_models.json"
    try:
        data = fetch_cached_json(url, cache_path, cache_timeout=3600, headers=headers)
        models = data.get("data", [])
        return models
    except Exception as e:
        print(f"Error fetching Groq models: {str(e)}")
        return []

@llm.hookimpl
def register_models(register):
    try:
        models = fetch_groq_models()
        for model in models:
            model_id = model["id"]
            full_model_id = f"groq/{model_id}"
            alias = DEFAULT_ALIASES.get(model_id)
            aliases = [alias] if alias else []
            if model_id in ["whisper-large-v3", "distil-whisper-large-v3-en"]:
                register(
                    GroqWhisper(
                        model_id=full_model_id,
                        model_name=model_id,
                    ),
                    aliases=aliases
                )
            else:
                register(
                    Groq(
                        model_id=full_model_id,
                        model_name=model_id,
                    ),
                    aliases=aliases
                )
    except Exception as e:
        print(f"Error registering Groq models: {str(e)}")

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

    def __init__(self, model_id, model_name):
        self.model_id = model_id
        self.model_name = model_name

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
        if not key:
            raise Exception("Groq API key not found. Please set the LLM_GROQ_KEY environment variable.")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        messages = self.build_messages(prompt, conversation)
        body = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            **{k: v for k, v in prompt.options.dict().items() if v is not None}
        }
        if prompt.options.response_format == "json":
            body["response_format"] = {"type": "json_object"}
            body["stream"] = False
        try:
            with httpx.stream("POST", url, json=body, headers=headers) as r:
                r.raise_for_status()
                if r.status_code == 200:
                    if stream and not body.get("response_format"):
                        for chunk in r.iter_lines():
                            if chunk.startswith("data: "):
                                chunk = chunk[6:].strip()
                                if chunk and chunk != "[DONE]":
                                    try:
                                        chunk_data = json.loads(chunk)
                                        if chunk_data["choices"][0]["delta"].get("content"):
                                            yield chunk_data["choices"][0]["delta"]["content"]
                                    except json.JSONDecodeError as e:
                                        print(f"Error decoding JSON chunk: {str(e)}")
                                        print(f"Raw chunk: {chunk}")
                    else:
                        response_json = r.json()
                        content = response_json["choices"][0]["message"]["content"]
                        if prompt.options.response_format == "json":
                            content = json.dumps(json.loads(content), indent=2)
                        yield content
                else:
                    yield f"Error: Unexpected response code {r.status_code}. Response: {r.text}"
        except httpx.HTTPError as e:
            print(f"Error during Groq API call: {str(e)}")
            yield f"Error: Unable to complete the request. {str(e)}"

class GroqWhisper(llm.Model):
    can_stream = False

    class Options(llm.Options):
        audio_file: str = Field(..., description="Path to the audio file to transcribe")
        prompt: Optional[str] = Field(default=None)
        response_format: Optional[str] = Field(default="json")
        temperature: Optional[float] = Field(default=0, ge=0, le=1)
        language: Optional[str] = Field(default=None)

    def __init__(self, model_id, model_name):
        self.model_id = model_id
        self.model_name = model_name

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "groq", "LLM_GROQ_KEY")
        if not key:
            raise Exception("Groq API key not found. Please set the LLM_GROQ_KEY environment variable.")
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {key}"}
        try:
            with open(prompt.options.audio_file, "rb") as audio:
                files = {"file": audio}
                data = {
                    "model": self.model_name,
                    **{k: v for k, v in prompt.options.dict().items() if v is not None and k != "audio_file"}
                }
                response = httpx.post(url, headers=headers, data=data, files=files)
                response.raise_for_status()
                if response.status_code == 200:
                    transcript = response.json().get("text", "No transcript available")
                    # Save transcript to file
                    file_name = os.path.splitext(os.path.basename(prompt.options.audio_file))[0]
                    output_file = f"{file_name}_transcript.txt"
                    with open(output_file, 'w') as f:
                        f.write(transcript)
                    print(f"Transcript saved to {output_file}")
                    yield transcript
                else:
                    yield f"Error: Unexpected response code {response.status_code}. Response: {response.text}"
        except FileNotFoundError:
            print(f"Error: Audio file '{prompt.options.audio_file}' not found.")
            yield "Error: Audio file not found."
        except httpx.HTTPError as e:
            print(f"Error during Whisper API call: {str(e)}")
            yield f"Error: Unable to transcribe the audio. {str(e)}"
        except IOError as e:
            print(f"Error saving transcript: {str(e)}")
            yield transcript  # Still yield the transcript even if saving fails
