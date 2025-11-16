import os
from collections import deque

import anthropic
import dotenv
import instructor
from ollama import Client
from openai import OpenAI
from pydantic import BaseModel

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

    def __init__(self, model: str = 'qwen3', provider=None):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if 'DEEPSEEK_API_KEY' in os.environ:
            self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        else:
            self.deepseek_api_key = None
        if 'ANTHROPIC_API_KEY' in os.environ:
            self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        else:
            self.anthropic_api_key = None
        if 'GOOGLE_API_KEY' in os.environ:
            self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        else:
            self.gemini_api_key = None
        self.llm = None
        self.model = model
        self.provider = provider
        self.chat_history = ChatHistory()
        self._set_active_model(model, provider)

    def reset_chat_history(self):
        """Reset the chat history"""
        self.chat_history = ChatHistory()

    def _setup_llm_client(self, provider='ollama'):
        """Setup the LLM client for the specified provider"""
        self.provider =  provider
        if provider == 'openai':
            self.llm = OpenAI(api_key=self.openai_api_key)
        elif provider == 'deepseek':
            self.llm = OpenAI(api_key=self.deepseek_api_key, base_url='https://api.deepseek.com/v1')
        elif provider == 'anthropic':
            self.llm = anthropic.Anthropic(api_key=self.anthropic_api_key)
        elif provider == 'google':
            self.llm = OpenAI(api_key=self.gemini_api_key, base_url='https://generativelanguage.googleapis.com/v1beta/openai/')
        elif provider == 'ollama' and ("OLLAMA_API_BASE" in os.environ):
            host = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
            self.llm = Client(host=host)

    def _fetch_provider_models(self, provider):
        """Fetch models from the specified provider, handling connection errors"""
        try:
            if provider == 'openai' and self.openai_api_key:
                return [m.id for m in self.llm.models.list().data]
            elif provider == 'deepseek' and self.deepseek_api_key:
                return [m.id for m in self.llm.models.list().data]
            elif provider == 'anthropic' and self.anthropic_api_key:
                return [m.id for m in self.llm.models.list().data]
            elif provider == 'google' and self.gemini_api_key:
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
         Get available models from the language model provider
        """
        if not self.llm:
            self._setup_llm_client(provider=self.provider)
        models = []
        models.extend(self._fetch_provider_models(self.provider))
        models.sort()
        return models

    def _find_model_provider(self, model: str):
        """Find which provider supports the requested model"""
        providers = ['openai', 'deepseek', 'anthropic', 'google', 'ollama']
        for provider in providers:
            try:
                # Setup client for this provider
                self._setup_llm_client(provider)
                # Get available models for this provider
                available_models = self._fetch_provider_models(provider)
                # Check if model is available
                if model in available_models:
                    return provider
            except Exception as e:
                print(f"Error checking provider {provider}: {e}")
                continue
        return None

    def _set_active_model(self, model: str, provider=None):
        # If provider is specified, use it directly
        if provider:
            self._setup_llm_client(provider)
            available_models = self._fetch_provider_models(provider)
            if model in available_models:
                self.model = model
                self.provider = provider
                return
            else:
                raise ValueError(f"Model {model} not found in provider {provider}")
        
        # Otherwise, search across all providers
        found_provider = self._find_model_provider(model)
        if found_provider:
            self.model = model
            self.provider = found_provider
            self._setup_llm_client(found_provider)
        else:
            # Try some fallbacks
            if 'gpt' in model.lower():
                # Try to use OpenAI
                self._setup_llm_client('openai')
                available_models = self._fetch_provider_models('openai')
                if available_models:
                    self.model = 'gpt-4o' if 'gpt-4o' in available_models else available_models[0]
                    self.provider = 'openai'
                else:
                    raise ValueError("OpenAI provider not available")
            elif 'gemini' in model.lower():
                # Try to use Google
                self._setup_llm_client('google')
                available_models = self._fetch_provider_models('google')
                if available_models:
                    self.model = 'gemini-2.5-flash' if 'gemini-2.5-flash' in available_models else available_models[0]
                    self.provider = 'google'
                else:
                    raise ValueError("Google provider not available")
            else:
                # Try ollama as last resort
                self._setup_llm_client('ollama')
                available_models = self._fetch_provider_models('ollama')
                if available_models:
                    self.model = available_models[0]
                    self.provider = 'ollama'
                else:
                    raise ValueError(
                        f"Model {model} not found in any provider.\n"
                        f"Please check if the model name is correct and the provider is properly configured."
                    )

    def get_response(self, question: str, context: str = None) -> str:
        """
        Get response from any supported model
        Args:
            question: str: question to ask
            context: str: question context to provide

        Returns: str: model's response
        """
        if not self.llm:
            self._setup_llm_client(provider=self.provider)
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

    def __init__(self, model: str = 'gpt-4o', provider: str = 'openai', retries=3):
        """
        Initialize the StructuredLangModel class with a language model.
        :param model:  Large Language Model to use.
        :param provider: Provider of the language model, e.g., 'openai', 'ollama', etc.
        :param retries: Number of retries to attempt.
        """
        super().__init__(model, provider=provider)
        self.retries = retries
        if 'gpt' in model:
            api_key = os.getenv('OPENAI_API_KEY')
            self.llm = instructor.from_openai(OpenAI(api_key=api_key))
        else:
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
