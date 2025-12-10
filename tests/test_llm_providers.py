"""Tests for LLM providers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codevid.llm import create_llm_provider, get_provider_for_name
from codevid.llm.anthropic_provider import AnthropicProvider
from codevid.llm.base import LLMError
from codevid.llm.ollama import OllamaProvider
from codevid.llm.openai_provider import OpenAIProvider
from codevid.models import ActionType, ParsedTest, TestStep
from codevid.models.project import LLMConfig, LLMProviderType


@pytest.fixture
def sample_test() -> ParsedTest:
    """Create a sample parsed test for testing."""
    return ParsedTest(
        name="test_login",
        file_path="test_login.py",
        steps=[
            TestStep(
                action=ActionType.NAVIGATE,
                target="https://example.com/login",
                description="Navigate to login page",
            ),
            TestStep(
                action=ActionType.TYPE,
                target="#email",
                value="user@example.com",
                description="Enter email",
            ),
            TestStep(
                action=ActionType.CLICK,
                target="#submit",
                description="Click submit",
            ),
        ],
        metadata={"docstring": "Test the login flow"},
    )


@pytest.fixture
def sample_llm_response() -> str:
    """Sample JSON response from LLM."""
    return json.dumps({
        "title": "Login Tutorial",
        "introduction": "In this tutorial, we'll learn how to log in.",
        "segments": [
            {"step_index": 0, "text": "First, navigate to the login page.", "timing_hint": 3.0},
            {"step_index": 1, "text": "Enter your email address.", "timing_hint": 2.5},
            {"step_index": 2, "text": "Click the submit button.", "timing_hint": 2.0},
        ],
        "conclusion": "You have successfully logged in!",
    })


class TestAnthropicProvider:
    @patch("anthropic.Anthropic")
    def test_init_creates_client(self, mock_anthropic_class):
        provider = AnthropicProvider(api_key="test-key")
        assert provider.provider_name == "anthropic"
        assert provider.model_name == "claude-sonnet-4-20250514"
        mock_anthropic_class.assert_called_once_with(api_key="test-key")

    @patch("anthropic.Anthropic")
    def test_custom_model(self, mock_anthropic_class):
        provider = AnthropicProvider(api_key="key", model="claude-3-opus-20240229")
        assert provider.model_name == "claude-3-opus-20240229"

    @patch("anthropic.Anthropic")
    @pytest.mark.asyncio
    async def test_generate_script(self, mock_anthropic_class, sample_test, sample_llm_response):
        # Setup mock response
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=sample_llm_response)]
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="test-key")
        script = await provider.generate_script(sample_test, {"app_name": "Test App"})

        assert script.title == "Login Tutorial"
        assert len(script.segments) == 3
        assert script.segments[0].text == "First, navigate to the login page."
        mock_client.messages.create.assert_called_once()

    @patch("anthropic.Anthropic")
    @pytest.mark.asyncio
    async def test_generate_script_with_code_block(self, mock_anthropic_class, sample_test, sample_llm_response):
        # Response wrapped in markdown code block
        wrapped_response = f"```json\n{sample_llm_response}\n```"

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=wrapped_response)]
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="test-key")
        script = await provider.generate_script(sample_test)

        assert script.title == "Login Tutorial"

    @patch("anthropic.Anthropic")
    @pytest.mark.asyncio
    async def test_api_error_raises_llm_error(self, mock_anthropic_class, sample_test):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")

        provider = AnthropicProvider(api_key="test-key")

        with pytest.raises(LLMError) as exc_info:
            await provider.generate_script(sample_test)

        assert "API call failed" in str(exc_info.value)
        assert exc_info.value.provider == "anthropic"


class TestOpenAIProvider:
    @patch("openai.OpenAI")
    def test_init_creates_client(self, mock_openai_class):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.provider_name == "openai"
        assert provider.model_name == "gpt-4o"
        mock_openai_class.assert_called_once_with(api_key="test-key", base_url=None)

    @patch("openai.OpenAI")
    def test_custom_base_url(self, mock_openai_class):
        provider = OpenAIProvider(api_key="key", base_url="https://custom.api")
        mock_openai_class.assert_called_once_with(api_key="key", base_url="https://custom.api")

    @patch("openai.OpenAI")
    @pytest.mark.asyncio
    async def test_generate_script(self, mock_openai_class, sample_test, sample_llm_response):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=sample_llm_response))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        script = await provider.generate_script(sample_test)

        assert script.title == "Login Tutorial"
        assert len(script.segments) == 3
        mock_client.chat.completions.create.assert_called_once()


class TestOllamaProvider:
    def test_init_defaults(self):
        provider = OllamaProvider()
        assert provider.provider_name == "ollama"
        assert provider.model_name == "llama3.2"
        assert provider._base_url == "http://localhost:11434"

    def test_custom_config(self):
        provider = OllamaProvider(model="mistral", base_url="http://server:11434/")
        assert provider.model_name == "mistral"
        assert provider._base_url == "http://server:11434"

    @pytest.mark.asyncio
    async def test_generate_script(self, sample_test, sample_llm_response):
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": sample_llm_response}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            provider = OllamaProvider()
            script = await provider.generate_script(sample_test)

            assert script.title == "Login Tutorial"
            assert len(script.segments) == 3

    @pytest.mark.asyncio
    async def test_connection_error(self, sample_test):
        import httpx

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")

            provider = OllamaProvider()

            with pytest.raises(LLMError) as exc_info:
                await provider.generate_script(sample_test)

            assert "Could not connect to Ollama" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_json_uses_defaults(self, sample_test):
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Not valid JSON response"}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            provider = OllamaProvider()
            script = await provider.generate_script(sample_test)

            # Should use defaults when JSON parsing fails
            assert script.title == "test_login"
            assert len(script.segments) == 3


class TestLLMFactory:
    def test_create_anthropic_provider(self):
        with patch("anthropic.Anthropic"):
            config = LLMConfig(provider=LLMProviderType.ANTHROPIC, api_key="key")
            provider = create_llm_provider(config)
            assert isinstance(provider, AnthropicProvider)

    def test_create_openai_provider(self):
        with patch("openai.OpenAI"):
            config = LLMConfig(provider=LLMProviderType.OPENAI, api_key="key")
            provider = create_llm_provider(config)
            assert isinstance(provider, OpenAIProvider)

    def test_create_ollama_provider(self):
        config = LLMConfig(provider=LLMProviderType.OLLAMA)
        provider = create_llm_provider(config)
        assert isinstance(provider, OllamaProvider)

    def test_get_provider_for_name(self):
        provider = get_provider_for_name("ollama", model="mistral")
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name == "mistral"

    def test_unknown_provider_raises_error(self):
        with pytest.raises(LLMError) as exc_info:
            get_provider_for_name("unknown_provider")

        assert "Unknown provider" in str(exc_info.value)
