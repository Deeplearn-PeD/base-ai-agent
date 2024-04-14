from openai import OpenAI
from ollama import Client
import ollama
import dotenv
import os

dotenv.load_dotenv()

class LangModel:
    """
    Interface to interact with language models
    """
    def __init__(self, model: str = 'gpt-4-0125-preview'):
        self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) if 'gpt' in model else Client(host='http://localhost:11434')
        self.model = model

    def get_response(self, question: str, context: str = None) -> str:
        if 'gpt' in self.model:
            return self.get_gpt_response(question, context)
        elif 'gemma' in self.model:
            return self.get_gemma_response(question, context)

    def get_gpt_response(self, question: str, context: str)->str:
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
            temperature=0.5,
            top_p=1
        )
        return response.choices[0].message.content

    def get_gemma_response(self, question: str, context: str) -> str:
        response = ollama.generate(
            model=self.model,
            system=context[:2048],
            prompt=question,
        )

        return response['response']
        # return '/n'.join([resp['response'] for resp in response ])


