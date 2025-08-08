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

    def __init__(self, model: str = 'gpt-4o', provider=''):
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

    def _setup_llm_client(self, provider='openai'):
        """Setup the LLM client for the specified provider"""
        if provider == 'openai':
            self.llm = OpenAI(api_key=self.openai_api_key)
        elif provider == 'deepseek':
            self.llm = OpenAI(api_key=self.deepseek_api_key, base_url='https://api.deepseek.com/v1')
        elif provider == 'anthropic':
            self.llm = anthropic.Anthropic(api_key=self.anthropic_api_key)
        elif provider == 'google':
            self.llm = OpenAI(api_key=self.gemini_api_key, base_url='https://generativelanguage.googleapis.com/v1beta/openai/')
        else:
            self.provider = 'ollama'
            host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            self.llm = Client(host=host)

    @property
    def available_models(self):
        """
         Get available models from the language model provider
        """
        if not self.llm:
            self._setup_llm_client(provider=self.provider)
        models = []
        if self.openai_api_key and self.provider == 'openai':
            models.extend([m.id for m in self.llm.models.list().data])
        if self.deepseek_api_key and self.provider == 'deepseek':
            models.extend([m.id for m in self.llm.models.list().data])
        if self.anthropic_api_key and self.provider == 'anthropic':
            models.extend([m.id for m in self.llm.models.list().data])
        if self.gemini_api_key and self.provider == 'google':
            models.extend([m.id for m in self.llm.models.list().data])
        # Ollama models
        try:
            host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            llm = Client(host=host)
            models.extend([m['name'].split(':')[0] for m in llm.list()['models']])
        except Exception as e:
            print(f"Error: {e}, unable to fetch Ollama models. ")
        models.sort()

        return models

    def _set_active_model(self, model: str, provider='openai'):
        available_model_names = self.available_models  # [m['name'].split(':')[0] for m in self.available_models]
        if 'gpt' in model:
            self.model = 'gpt-4o'
        if 'gemini' in model:
            self.model = 'gemini-2.5-flash'
        elif model in available_model_names:
            self.model = model
        else:
            raise ValueError(
                f"Model {model} not supported.\nAvailable models: {[m for m in self.available_models]}")
            self.model = "llama3.2"

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
        if 'gpt' in self.model or 'gemini' in self.model:
            return self.get_gpt_response(question, context)
        else:
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
                    base_url=os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434/v1'),
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

    def get_response(self, question: str, context: str = "", response_model: BaseModel = None) -> str:
        """
        Get response from any supported model

        :param question: question to ask
        :param context: question context to provide
        :param response_model: response model to use
        :return: model's response in JSON format
        """
        msg = {'role': 'user', 'content': context + '\n\n' + question}
        self.chat_history.enqueue(msg)
        messages = self.chat_history.get_all()
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=response_model,
            max_retries=self.retries
        )
        self.chat_history.enqueue(response)

        return response
