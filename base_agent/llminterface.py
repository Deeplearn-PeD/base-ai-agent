import os
from collections import deque

import anthropic
import dotenv
import instructor
from ollama import Client
from openai import OpenAI
from pydantic import BaseModel
from collections import defaultdict

dotenv.load_dotenv()


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
    supported_API_KEYS = {
        'openai': 'OPENAI_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'ollama': 'OLLAMA_API_BASE',
        'qwen': 'DASHSCOPE_API_KEY'
    }

    def __init__(self, model: str = 'qwen3', provider=None):
        for provider, env_var in self.supported_API_KEYS.items():
            self.keys = {}
            if env_var in os.environ:
                self.keys[provider] = os.getenv(env_var)
            else:
                self.keys[provider] = None

        self.llm = None
        self.model = model
        self._available_models = []
        self.available_models
        self.provider_models = defaultdict(lambda:[])
        self.provider = None
        self.chat_history = ChatHistory()
        self._set_active_model(model)

    def reset_chat_history(self):
        """Reset the chat history"""
        self.chat_history = ChatHistory()

    def _setup_llm_client(self, provider: str='ollama', key: str=''):
        """Setup the LLM client for the specified provider"""
        self.provider =  provider
        if provider == 'openai':
            self.llm = OpenAI(api_key=self.keys[provider])
        elif provider == 'deepseek':
            self.llm = OpenAI(api_key=self.keys[provider], base_url='https://api.deepseek.com/v1')
        elif provider == 'anthropic':
            self.llm = anthropic.Anthropic(api_key=self.keys[provider])
        elif provider == 'google':
            self.llm = OpenAI(api_key=self.keys[provider], base_url='https://generativelanguage.googleapis.com/v1beta/openai/')
        elif provider == 'qwen':
            self.llm = OpenAI(api_key=self.keys[provider],base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        elif provider == 'ollama' and ("OLLAMA_API_BASE" in os.environ):
            host = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
            self.llm = Client(host=host)

    def _fetch_provider_models(self, provider):
        """Fetch models from the specified provider, handling connection errors"""
        try:
            if provider == 'openai' and self.keys[provider]:
                return [m.id for m in self.llm.models.list().data]
            elif provider == 'deepseek' and self.keys[provider]:
                return [m.id for m in self.llm.models.list().data]
            elif provider == 'anthropic' and self.keys[provider]:
                return [m.id for m in self.llm.models.list().data]
            elif provider == 'google' and self.keys[provider]:
                return [m.id for m in self.llm.models.list().data]
            elif provider == 'ollama':
                host = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
                llm = Client(host=host)
                return [m['name'].split(':')[0] for m in llm.list()['models']]
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
            if model in self.provider_models[provider]:
                return provider
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
                  f"Setting up {self.available_models[0]} model."
            )
            self.model = self.available_models[0]
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

    def __init__(self, model: str = 'gpt-4o', retries=3):
        """
        Initialize the StructuredLangModel class with a language model.
        :param model:  Large Language Model to use.
        :param retries: Number of retries to attempt.
        """
        super().__init__(model)
        self.retries = retries
        # Setup instructor based on the found provider
        if self.provider in ['openai', 'deepseek', 'google']:
            if self.provider == 'openai':
                api_key = self.openai_api_key
                base_url = None
            elif self.provider == 'deepseek':
                api_key = self.deepseek_api_key
                base_url = 'https://api.deepseek.com/v1'
            elif self.provider == 'google':
                api_key = self.gemini_api_key
                base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'
            
            self.llm = instructor.from_openai(
                OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            )
        else:  # ollama
            self.llm = instructor.from_openai(
                OpenAI(
                    base_url=os.getenv('OLLAMA_API_BASE', 'http://127.0.0.1:11434/v1'),
                    api_key=os.getenv('OLLAMA_API_KEY', 'ollama')
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
