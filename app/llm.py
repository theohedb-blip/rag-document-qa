"""Embedding and answer providers. Swap them out (e.g. with fakes in tests) via the two Protocols."""

from typing import Protocol

from app.config import Settings


class MissingAPIKey(RuntimeError):
    pass


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class AnswerModel(Protocol):
    def answer(self, system: str, prompt: str) -> str: ...


class OpenAIEmbedder:
    batch_size = 100

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            response = self._client.embeddings.create(model=self._model, input=texts[start : start + self.batch_size])
            vectors.extend(item.embedding for item in response.data)
        return vectors


class OpenAIAnswerModel:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def answer(self, system: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


class AnthropicAnswerModel:
    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def answer(self, system: str, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},  # short grounded answers don't need deep reasoning
        )
        return "".join(block.text for block in response.content if block.type == "text")


def make_embedder(settings: Settings) -> Embedder:
    _require(settings.openai_api_key, "OPENAI_API_KEY")
    return OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)


def make_answer_model(settings: Settings) -> AnswerModel:
    if settings.llm_provider == "anthropic":
        _require(settings.anthropic_api_key, "ANTHROPIC_API_KEY")
        return AnthropicAnswerModel(settings.anthropic_api_key, settings.anthropic_model)
    _require(settings.openai_api_key, "OPENAI_API_KEY")
    return OpenAIAnswerModel(settings.openai_api_key, settings.openai_chat_model)


def _require(value: str, name: str) -> None:
    if not value:
        raise MissingAPIKey(f"{name} is not set. Copy .env.example to .env and add your key.")
