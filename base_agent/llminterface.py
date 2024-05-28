from openai import OpenAI
from ollama import Client
import ollama
import dotenv
import os
from collections import deque
from typing import List, Dict

dotenv.load_dotenv()

class ChatHistory:
    """
    The ChatHistory class is a FIFO queue that keeps track of chat history.

    Attributes:
        queue (collections.deque): A deque object that stores the chat history.
    """

    def __init__(self, max_size=1000):
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

    def get_all(self):
        """
        Return all items in the queue as a list without removing them from the queue.

        Returns:
            list: A list of all items in the queue.
        """
        return list(self.queue)
class LangModel:
    """
    Interface to interact with language models
    """

    def __init__(self, model: str = 'gpt-4o'):
        if 'gpt' in model:
            api_key = os.getenv('OPENAI_API_KEY')
            self.llm = OpenAI(api_key=api_key)
        else:
            host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            self.llm = Client(host=host)

        self.available_models: List = ollama.list()['models']
        self.model = "llama3"
        self.chat_history = ChatHistory()
        self._set_active_model(model)

    def _set_active_model(self, model: str):
        if model in [m['name'].split(':')[0] for m in self.available_models]:
            self.model = model
        elif 'gpt' in model:
            self.model = 'gpt-4o'
        else:
            raise ValueError(f"Model {model} not supported.\nAvailable models: {[m['name'] for m in self.available_models]}")
            self.model = "llama3"


    def get_response(self, question: str, context: str = None) -> str:
        if 'gpt' in self.model:
            return self.get_gpt_response(question, context)
        else:
            return self.get_ollama_response(question, context)


    def get_gpt_response(self, question: str, context: str) -> str:
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'system',
                    'content': context
                },
                {
                    'role': 'user',
                    'content': question
                }
            ],
            max_tokens=100,
            temperature=0.1,
            top_p=1
        )
        return response.choices[0].message.content

    def get_ollama_response(self, question: str, context: str) -> str:
        """
        Get response from any Ollama supported model
        :param question: question to ask
        :param context: context to provide
        :return: model's response
        """
        msg = {
            'role': 'user',
            'content': context + '\n\n' + question
        }
        self.chat_history.enqueue(msg)
        messages = self.chat_history.get_all()
        response = self.llm.chat(
            model=self.model,
            messages=messages,
            options={'temperature': 0}
        )
        self.chat_history.enqueue(response['message'])

        return response['message']['content']
