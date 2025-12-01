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

    def _setup_llm_client(self, provider: str='ollama', key: str=''):
        """Setup the LLM client for the specified provider using config.yml"""
        self.provider = provider
        
        # Get provider configuration from config.yml
        provider_config = {}
        if CONFIG and 'providers' in CONFIG:
            provider_config = CONFIG['providers'].get(provider, {})
        
        # Determine base_url and default_model from config, with fallbacks
        base_url = provider_config.get('base_url')
        default_model = provider_config.get('default_model')
        
        if provider == 'openai':
            self.llm = OpenAI(api_key=self.keys[provider], base_url=base_url)
        elif provider == 'deepseek':
            if not base_url:
                base_url = 'https://api.deepseek.com/v1'
            self.llm = OpenAI(api_key=self.keys[provider], base_url=base_url)
        elif provider == 'anthropic':
            self.llm = anthropic.Anthropic(api_key=self.keys[provider])
        elif provider == 'google':
            if not base_url:
                base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'
            self.llm = OpenAI(api_key=self.keys[provider], base_url=base_url)
        elif provider == 'qwen':
            if not base_url:
                base_url = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
            self.llm = OpenAI(api_key=self.keys[provider], base_url=base_url)
        elif provider == 'ollama':
            # For ollama, base_url is used as host
            if not base_url:
                base_url = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
            else:
                # If base_url is set in config, it should be used
                # But environment variable may override
                env_host = os.getenv('OLLAMA_API_BASE')
                if env_host:
                    base_url = env_host
            self.llm = Client(host=base_url)

    def _fetch_provider_models(self, provider):
        """Fetch models from the specified provider, handling connection errors"""
        try:
            # Helper to get a temporary client for the provider
            def get_temp_client():
                # If self.llm is already initialized for this provider, reuse it
                if self.llm is not None and self.provider == provider:
                    return self.llm
                # Otherwise create a new client based on provider
                provider_config = {}
                if CONFIG and 'providers' in CONFIG:
                    provider_config = CONFIG['providers'].get(provider, {})
                
                api_key = self.keys.get(provider)
                if not api_key:
                    return None
                
                if provider in ['openai', 'deepseek', 'google', 'qwen']:
                    base_url = provider_config.get('base_url')
                    if provider == 'deepseek' and not base_url:
                        base_url = 'https://api.deepseek.com/v1'
                    elif provider == 'google' and not base_url:
                        base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'
                    elif provider == 'qwen' and not base_url:
                        base_url = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
                    return OpenAI(api_key=api_key, base_url=base_url)
                elif provider == 'anthropic':
                    return anthropic.Anthropic(api_key=api_key)
                else:
                    return None
            
            if provider == 'ollama':
                host = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
                llm = Client(host=host)
                return [m['name'].split(':')[0] for m in llm.list()['models']]
            
            # For other providers, ensure we have a key
            if not self.keys.get(provider):
                return []
            
            temp_client = get_temp_client()
            if temp_client is None:
                return []
            
            # Fetch models using the appropriate method
            if provider == 'anthropic':
                # Anthropic uses models.list() as well
                return [m.id for m in temp_client.models.list().data]
            else:
                # OpenAI-compatible providers
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
        # Determine which method to use based on provider
        if self.provider in ['openai', 'deepseek', 'google']:
            return self.get_gpt_response(question, context)
        elif self.provider == 'anthropic':
            return self.get_anthropic_response(question, context)
        else:  # ollama
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
        # If no model is specified, try to get default from config for structured use
        if model is None:
            # Look for a suitable default model for structured output
            if CONFIG and 'providers' in CONFIG:
                # Prefer openai's default model
                openai_config = CONFIG['providers'].get('openai', {})
                model = openai_config.get('default_model', 'gpt-4o')
            else:
                model = 'gpt-4o'
        
        super().__init__(model)
        self.retries = retries
        
        # Setup instructor based on the found provider using config
        if self.provider in ['openai', 'deepseek', 'google', 'qwen']:
            # Get provider configuration
            provider_config = {}
            if CONFIG and 'providers' in CONFIG:
                provider_config = CONFIG['providers'].get(self.provider, {})
            
            base_url = provider_config.get('base_url')
            api_key = self.keys.get(self.provider)
            
            # Create OpenAI client with config
            client_kwargs = {'api_key': api_key}
            if base_url:
                client_kwargs['base_url'] = base_url
            
            self.llm = instructor.from_openai(
                OpenAI(**client_kwargs)
            )
        else:  # ollama or others
            # For ollama, use the host from config or environment
            host = os.getenv('OLLAMA_API_BASE', 'http://127.0.0.1:11434')
            if CONFIG and 'providers' in CONFIG and 'ollama' in CONFIG['providers']:
                ollama_config = CONFIG['providers']['ollama']
                config_host = ollama_config.get('base_url')
                if config_host:
                    host = config_host
            
            self.llm = instructor.from_openai(
                OpenAI(
                    base_url=host + '/v1' if not host.endswith('/v1') else host,
                    api_key='ollama'  # Dummy key for ollama
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
