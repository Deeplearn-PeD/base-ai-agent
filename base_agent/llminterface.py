import os
import yaml
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Optional, Union, Any, Dict

import dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart, TextPart

dotenv.load_dotenv()

# Load configuration from config.yml
CONFIG_PATH = Path(__file__).parent / "config.yml"
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    CONFIG = {}
except Exception as e:
    print(f"Warning: Could not load config.yml: {e}")
    CONFIG = {}


class ChatHistory:
    """
    The ChatHistory class is a FIFO queue that keeps track of chat history.
    This is a non-persistent memory that will last only for a chat session.
    Attributes:
        queue (collections.deque): A deque object that stores the chat history.
    """

    def __init__(self, max_size=1000):
        """
        Initialize the ChatHistory class with a maximum size.

        Args:
            max_size (int): The maximum size of the queue. Defaults to 1000.
        """
        self.queue: deque[ModelMessage] = deque(maxlen=max_size)

    def enqueue(self, item: Union[ModelMessage, Dict[str, Any]]):
        """
        Add a message to the end of the queue.
        Converts dict messages (OpenAI format) to Pydantic AI ModelMessage if needed.

        Args:
            item: The message to be added to the queue.
        """
        if isinstance(item, dict):
            role = item.get('role')
            content = item.get('content')
            if role == 'user':
                self.queue.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == 'assistant':
                self.queue.append(ModelResponse(parts=[TextPart(content=content)]))
        else:
            self.queue.append(item)

    def dequeue(self):
        """
        Remove and return a message from the front of the queue.

        Returns:
            The message removed from the front of the queue. If the queue is empty, returns None.
        """
        if len(self.queue) == 0:
            return None
        return self.queue.popleft()

    def get_all(self) -> List[ModelMessage]:
        """
        Return all items in the queue as a list without removing them from the queue.

        Returns:
            list: A list of all items in the queue.
        """
        return list(self.queue)


class LangModel:
    """
    Base class for language model interfaces using Pydantic AI
    """
    # Load provider configuration from config.yml
    if CONFIG and 'providers' in CONFIG:
        provider_configs = CONFIG['providers']
        supported_API_KEYS = {
            provider: config.get('api_key_env_var', '')
            for provider, config in provider_configs.items()
        }
    else:
        supported_API_KEYS = {}

    def __init__(self, model: str = None, provider: str = None):
        # Initialize keys from environment variables using supported_API_KEYS
        self.keys = {}
        for provider_name, env_var in self.supported_API_KEYS.items():
            if env_var in os.environ:
                self.keys[provider_name] = os.getenv(env_var)
            else:
                self.keys[provider_name] = None

        self.agent: Optional[Agent] = None
        self._available_models = []
        self.provider_models = defaultdict(list)
        self.provider = provider
        self.chat_history = ChatHistory()

        # If no model is specified, try to get default model from config.yml
        if model is None:
            for provider_name, config in CONFIG.get('providers', {}).items():
                if provider_name in self.keys and self.keys[provider_name]:
                    model = config.get('default_model')
                    if model:
                        self.provider = provider_name
                        break
            if model is None:
                model = 'qwen3'

        self.model = model
        # Pre-fetch models to populate provider_models
        _ = self.available_models
        self._set_active_model(model)

    def reset_chat_history(self):
        """Reset the chat history"""
        self.chat_history = ChatHistory()

    def add_provider(self, name: str, default_model: str, base_url: str,
                     api_key_env_var: str, client_type: str):
        """Register a new provider at runtime.

        Args:
            name: Provider name (e.g. 'openai', 'ollama').
            default_model: Default model identifier for this provider.
            base_url: API base URL for this provider.
            api_key_env_var: Environment variable name holding the API key.
            client_type: One of 'openai', 'anthropic', or 'ollama'.
        """
        config = {
            'default_model': default_model,
            'base_url': base_url,
            'api_key_env_var': api_key_env_var,
            'client_type': client_type,
        }

        CONFIG.setdefault('providers', {})[name] = config
        LangModel.supported_API_KEYS[name] = api_key_env_var
        self.keys[name] = os.getenv(api_key_env_var) if api_key_env_var in os.environ else None

        self._available_models = []

    def _get_model_instance(self, provider: str, model_name: str) -> Model:
        """Create a Pydantic AI Model instance based on provider and config"""
        provider_config = CONFIG.get('providers', {}).get(provider, {})
        base_url = provider_config.get('base_url')
        api_key = self.keys.get(provider)

        if provider == 'anthropic':
            return AnthropicModel(model_name, api_key=api_key)

        # Default to OpenAIModel for OpenAI-compatible providers
        # Most providers in config.yml (OpenAI, DeepSeek, Google, Qwen, Ollama)
        # are used via their OpenAI-compatible endpoints.
        if provider == 'ollama' and base_url and not base_url.endswith('/v1'):
            base_url = f"{base_url.rstrip('/')}/v1"

        return OpenAIModel(model_name, api_key=api_key or 'dummy', base_url=base_url)

    def _setup_llm_client(self, provider: str = 'ollama'):
        """Setup the Pydantic AI Agent for the specified provider"""
        self.provider = provider
        model_instance = self._get_model_instance(provider, self.model)
        self.agent = Agent(model_instance)

    def _fetch_provider_models(self, provider):
        """Fetch models from the specified provider using Pydantic AI compatible logic"""
        try:
            if provider == 'ollama':
                import httpx
                host = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
                resp = httpx.get(f"{host}/api/tags", timeout=2.0)
                if resp.status_code == 200:
                    return [m['name'].split(':')[0] for m in resp.json().get('models', [])]
                return []

            # For OpenAI-compatible providers, use httpx directly
            provider_config = CONFIG.get('providers', {}).get(provider, {})
            base_url = provider_config.get('base_url')
            api_key = self.keys.get(provider)

            if not api_key:
                return []

            import httpx
            headers = {"Authorization": f"Bearer {api_key}"}
            list_url = f"{base_url.rstrip('/')}/models"
            resp = httpx.get(list_url, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                return [m['id'] for m in resp.json().get('data', [])]

        except Exception as e:
            print(f"Error fetching models from {provider}: {e}")
        return []

    @property
    def available_models(self):
        """
        Get available models from all configured providers
        """
        if self._available_models:
            return self._available_models

        models = []
        self.provider_models = {}

        # Always try ollama
        try:
            ollama_models = self._fetch_provider_models('ollama')
            if ollama_models:
                self.provider_models['ollama'] = ollama_models
                models.extend(ollama_models)
        except Exception:
            pass

        # Add models from other providers if their API keys are available
        for provider, key in self.keys.items():
            if key and provider != 'ollama':
                mods = self._fetch_provider_models(provider)
                if mods:
                    self.provider_models[provider] = mods
                    models.extend(mods)

        # Remove duplicates and sort
        models = list(set(models))
        models.sort()
        self._available_models = models
        return models

    def _find_model_provider(self, model: str):
        """Find which provider supports the requested model"""
        for provider in self.keys.keys():
            if model in self.provider_models.get(provider, []):
                return provider
        return None

    def _set_active_model(self, model: str):
        found_provider = self._find_model_provider(model)
        if found_provider:
            self.model = model
            self._setup_llm_client(found_provider)
        else:
            fallback_model = 'qwen3' if 'qwen3' in self.available_models else \
                            (self.available_models[-1] if self.available_models else model)
            print(f"Model {model} not found. Using fallback: {fallback_model}")
            self.model = fallback_model
            provider = self._find_model_provider(self.model) or 'ollama'
            self._setup_llm_client(provider)

    def get_response(self, question: str, context: str = None) -> str:
        """
        Get response from the agent
        Args:
            question: str: question to ask
            context: str: system context to provide

        Returns: str: model's response
        """
        if not self.agent:
            self._setup_llm_client(self.provider or 'ollama')

        import asyncio

        async def _run():
            history = self.chat_history.get_all()

            result = await self.agent.run(
                question,
                system_prompt=context,
                message_history=history
            )

            for msg in result.new_messages():
                self.chat_history.enqueue(msg)

            return result.data

        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(_run())

class StructuredLangModel(LangModel):
    """
    Interface to interact with language models using structured query models
    """

    def __init__(self, model: str = None, retries=3):
        """
        Initialize the StructuredLangModel class with a language model.
        :param model:  Large Language Model to use.
        :param retries: Number of retries to attempt.
        """
        if model is None:
            if CONFIG and 'providers' in CONFIG:
                openai_config = CONFIG['providers'].get('openai', {})
                model = openai_config.get('default_model', 'gpt-4o')
            else:
                model = 'gpt-4o'

        super().__init__(model)
        self.retries = retries

    def get_response(self, question: str, context: str = "", response_model: Optional[type[BaseModel]] = None) -> Any:
        """
        Get response from any supported model using Pydantic AI's result_type

        :param question: question to ask
        :param context: system context to provide
        :param response_model: response model to use (Pydantic BaseModel)
        :return: model's response in the specified format
        """
        if not self.agent:
            self._setup_llm_client(self.provider or 'ollama')

        import asyncio
        import nest_asyncio
        nest_asyncio.apply()

        async def _run():
            history = self.chat_history.get_all()

            result = await self.agent.run(
                question,
                system_prompt=context,
                message_history=history,
                result_type=response_model
            )

            for msg in result.new_messages():
                self.chat_history.enqueue(msg)

            return result.data

        return asyncio.run(_run())
