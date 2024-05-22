from openai import OpenAI
from ollama import Client
import ollama
import dotenv
import os
from typing import List, Dict

dotenv.load_dotenv()


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
        elif self.model == 'gemma':
            self.model = 'gemma'
            return self.get_gemma_response(question, context)
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

    def get_gemma_response(self, question: str, context: str) -> str:
        response = ollama.generate(
            model=self.model,
            system=context,
            prompt=question,
        )

        return response['response']
        # return '/n'.join([resp['response'] for resp in response ])

    def get_ollama_response(self, question: str, context: str) -> str:
        """
        Get response from any Ollama supported model
        :param question: question to ask
        :param context: context to provide
        :return: model's response
        """
        response = self.llm.generate(
            model=self.model,
            system=context,
            prompt=question,
        )

        return response['response']
        # return '/n'.join([resp['response'] for resp in response ])
