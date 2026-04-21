import os
import yaml
from collections import deque, defaultdict
from pathlib import Path

import anthropic
import dotenv
import instructor
from ollama import Client
from openai import OpenAI
from pydantic import BaseModel

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
    This is a non-persistent memory that will last only for a chat session..
    Attributes:
        queue (collections.deque): A deque object that stores the chat history.
    """

    def __init__(self, max_size=1000, session_id=None):
        """
        Initialize the ChatHistory class with a maximum size.

        Args:
            max_size (int): The maximum size of the queue. Defaults to 1000.
        """
        self.queue = deque(maxlen=max_size)

    def enqueue(self, item):
        """
        Add a message to the end of the queue.

        Args:
            item: The message to be added to the queue.
        """
        self.queue.append(item)

    def dequeue(self):
        """
        Remove and return a message  from the front of the queue.

        Returns:
            The message removed from the front of the queue. If the queue is empty, returns None.
        """
        if len(self.queue) == 0:
            return None
        return self.queue.popleft()

    def _manage_chat_history(self, message: dict, response: dict):
        self.chat_history.enqueue(message)
        self.chat_history.enqueue(response)

    def get_all(self):
        """
        Return all items in the queue as a list without removing them from the queue.

        Returns:
            list: A list of all items in the queue.
        """
        return list(self.queue)


class LangModel:
    """
    Base class for language model interfaces
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

    def __init__(self, model: str = None, provider=None):
        # Initialize keys from environment variables using supported_API_KEYS
        self.keys = {}
        for provider_name, env_var in self.supported_API_KEYS.items():
            if env_var in os.environ:
                self.keys[provider_name] = os.getenv(env_var)
            else:
                self.keys[provider_name] = None
        
        self.llm = None
        self._available_models = []
        self.provider_models = defaultdict(lambda: [])
        self.provider = None
        self.chat_history = ChatHistory()
        
        # If no model is specified, try to get default model from config.yml
        if model is None:
            # Try to get default model from the first provider that has a key
            for provider_name, config in CONFIG.get('providers', {}).items():
                if provider_name in self.keys and self.keys[provider_name]:
                    model = config.get('default_model')
                    if model:
                        break
            # If still None, use a fallback
            if model is None:
                model = 'qwen3'
        
        self.model = model
        self.available_models  # This triggers fetching models
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

    def _setup_llm_client(self, provider: str = 'ollama'):
        """Setup the LLM client for the specified provider using config.yml"""
        self.provider = provider
        provider_config = CONFIG.get('providers', {}).get(provider, {})
        client_type = provider_config.get('client_type', 'openai')
        base_url = provider_config.get('base_url')
        api_key = self.keys.get(provider)

        if client_type == 'openai':
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        elif client_type == 'anthropic':
            self.llm = anthropic.Anthropic(api_key=api_key)
        elif client_type == 'ollama':
            env_var = provider_config.get('api_key_env_var')
            if env_var:
                env_val = os.getenv(env_var)
                if env_val:
                    base_url = env_val
            if not base_url:
                base_url = 'http://localhost:11434'
            self.llm = Client(host=base_url)

    def _fetch_provider_models(self, provider):
        """Fetch models from the specified provider, handling connection errors"""
        try:
            provider_config = CONFIG.get('providers', {}).get(provider, {})
            client_type = provider_config.get('client_type', 'openai')
            base_url = provider_config.get('base_url')
            api_key = self.keys.get(provider)

            if client_type == 'ollama':
                env_var = provider_config.get('api_key_env_var')
                if env_var:
                    env_val = os.getenv(env_var)
                    if env_val:
                        base_url = env_val
                if not base_url:
                    base_url = 'http://localhost:11434'
                llm = Client(host=base_url)
                return [m['name'].split(':')[0] for m in llm.list()['models']]

            if not api_key:
                return []

            if client_type == 'openai':
                temp_client = OpenAI(api_key=api_key, base_url=base_url)
            elif client_type == 'anthropic':
                temp_client = anthropic.Anthropic(api_key=api_key)
            else:
                return []

            return [m.id for m in temp_client.models.list().data]

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
        # Always include ollama models
        try:
            ollama_models = self._fetch_provider_models('ollama')
            self.provider_models['ollama'] = ollama_models
            models.extend(ollama_models)
        except Exception:
            pass
        
        # Add models from other providers if their API keys are available
        for provider, key in self.keys.items():
            if key:
                mods = self._fetch_provider_models(provider)
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
            x = self.available_models
            try:
                if model in self.provider_models[provider]:
                    return provider
            except KeyError:
                print(f"Provider {provider} not found. Skipping")
        else:
            print(f"Model {model} not found in any provider")



    def _set_active_model(self, model: str):
        # Search across all available providers
        found_provider = self._find_model_provider(model)
        if model in self.available_models:
            self.model = model
            self._setup_llm_client(found_provider)
        else:
            print(f"Model {model} not found in any available provider.\n"
                f"Available models: {self.available_models}\n"
                  f"Setting up {'qwen3' if 'qwen3' in self.available_models else self.available_models[-1]} model."
            )
            self.model = 'qwen3' if 'qwen3' in self.available_models else self.available_models[-1]
            self._setup_llm_client(found_provider)

    def get_response(self, question: str, context: str = None) -> str:
        """
        Get response from any supported model
        Args:
            question: str: question to ask
            context: str: question context to provide

        Returns: str: model's response
        """
        if not self.llm:
            self._setup_llm_client(self.provider)

        provider_config = CONFIG.get('providers', {}).get(self.provider, {})
        client_type = provider_config.get('client_type', 'openai')

        if client_type == 'openai':
            return self.get_gpt_response(question, context)
        elif client_type == 'anthropic':
            return self.get_anthropic_response(question, context)
        elif client_type == 'ollama':
            return self.get_ollama_response(question, context)

    def get_gpt_response(self, question: str, context: str) -> str:
        msg = {'role': 'user', 'content': question}
        self.chat_history.enqueue(msg)
        history = self.chat_history.get_all()
        response = self.llm.chat.completions.create(model=self.model,
                                 messages=[{'role': 'system', 'content': context}] + history,
                                 # max_tokens=1000,
                                 # temperature=0.1,
                                 # top_p=1
                                 )
        resp_msg = {'role': 'assistant', 'content': response.choices[0].message.content}
        self.chat_history.enqueue(resp_msg)
        return response.choices[0].message.content

    def get_anthropic_response(self, question: str, context: str) -> str:
        """
        Get response from Anthropic models
        :param question: question to ask
        :param context: context to provide
        :return: model's response
        """
        msg = {'role': 'user', 'content': context + '\n\n' + question}
        self.chat_history.enqueue(msg)
        messages = self.chat_history.get_all()
        
        # Convert messages to Anthropic format
        system_prompt = None
        anthropic_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                anthropic_messages.append({"role": msg['role'], "content": msg['content']})
        
        response = self.llm.messages.create(
            model=self.model,
            messages=anthropic_messages,
            system=system_prompt,
            max_tokens=1000
        )
        resp_msg = {'role': 'assistant', 'content': response.content[0].text}
        self.chat_history.enqueue(resp_msg)
        
        return response.content[0].text

    def get_ollama_response(self, question: str, context: str) -> str:
        """
        Get response from any Ollama supported model
        :param question: question to ask
        :param context: context to provide
        :return: model's response
        """
        msg = {'role': 'user', 'content': context + '\n\n' + question}
        self.chat_history.enqueue(msg)
        messages = self.chat_history.get_all()
        response = self.llm.chat(model=self.model, messages=messages, options={'temperature': 0})
        self.chat_history.enqueue(response['message'])

        return response['message']['content']


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
        
        provider_config = CONFIG.get('providers', {}).get(self.provider, {})
        client_type = provider_config.get('client_type', 'openai')
        base_url = provider_config.get('base_url')
        api_key = self.keys.get(self.provider)

        if client_type == 'openai':
            client_kwargs = {'api_key': api_key}
            if base_url:
                client_kwargs['base_url'] = base_url
            self.llm = instructor.from_openai(OpenAI(**client_kwargs))
        elif client_type == 'anthropic':
            self.llm = instructor.from_anthropic(
                anthropic.Anthropic(api_key=api_key)
            )
        elif client_type == 'ollama':
            env_var = provider_config.get('api_key_env_var')
            if env_var:
                env_val = os.getenv(env_var)
                if env_val:
                    base_url = env_val
            if not base_url:
                base_url = 'http://localhost:11434'
            self.llm = instructor.from_openai(
                OpenAI(
                    base_url=base_url + '/v1' if not base_url.endswith('/v1') else base_url,
                    api_key='ollama'
                ),
                mode=instructor.Mode.JSON,
                stream=False
            )

    def reset_chat_history(self):
        """
        Reset the chat history.
        """
        self.chat_history = ChatHistory()

    def get_response(self, question: str, context: str = "", response_model: BaseModel = None):
        """
        Get response from any supported model

        :param question: question to ask
        :param context: question context to provide
        :param response_model: response model to use
        :return: model's response in the specified format
        """
        msg = {'role': 'user', 'content': context + '\n\n' + question}
        self.chat_history.enqueue(msg)
        messages = self.chat_history.get_all()
        
        # Use instructor to get structured response
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=response_model,
            max_retries=self.retries
        )
        
        # The response is already the structured model instance
        resp_msg = {'role': 'assistant', 'content': str(response)}
        self.chat_history.enqueue(resp_msg)

        return response
