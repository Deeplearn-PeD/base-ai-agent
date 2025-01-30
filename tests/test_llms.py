import unittest
from unittest.mock import patch, MagicMock
from base_agent.llminterface import LangModel, StructuredLangModel
from pydantic import BaseModel, Field
from typing import List
import json


class TestLangModel(unittest.TestCase):
    @patch('base_agent.llminterface.OpenAI')
    @patch('base_agent.llminterface.Client')
    @patch('base_agent.llminterface.ollama')
    def test_init_with_gpt_model(self, mock_ollama, mock_client, mock_openai):
        mock_ollama.list.return_value = {'models': [{'name': 'gpt-4-turbo'}]}
        LangModel('gpt-4o')
        mock_openai.assert_called_once()

    def test_init_with_non_gpt_model(self):
        LangModel(model='deepseek-chat', provider='deepseek')


    @patch('base_agent.llminterface.OpenAI')
    @patch('base_agent.llminterface.Client')
    @patch('base_agent.llminterface.ollama')
    def test_init_with_unsupported_model(self, mock_ollama, mock_client, mock_openai):
        mock_ollama.list.return_value = {'models': [{'name': 'llama3.1'}]}
        with self.assertRaises(ValueError):
            LangModel('unsupported_model')

    @patch('base_agent.llminterface.LangModel.get_gpt_response')
    def test_get_response_with_gpt_model(self, mock_get_gpt_response):
        lm = LangModel('gpt-4o')
        lm.get_response('question', 'context')
        mock_get_gpt_response.assert_called_once_with('question', 'context')


    @patch('base_agent.llminterface.LangModel.get_ollama_response')
    def test_get_response_with_llama_model(self, mock_get_ollama_response):
        lm = LangModel('llama3.2')
        lm.get_response('question', 'context')
        mock_get_ollama_response.assert_called_once_with('question', 'context')


class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


class TestStructuredLangModel_GPT(unittest.TestCase):
    def test_get_strutured_output(self):
        slm = StructuredLangModel()
        response = slm.get_response('Tell me about Harry Potter', '', response_model=Character)
        expected = """
{
  "name": "Harry James Potter",
  "age": 37,
  "fact": [
    "He is the chosen one.",
    "He has a lightning-shaped scar on his forehead.",
    "He is the son of James and Lily Potter.",
    "He attended Hogwarts School of Witchcraft and Wizardry.",
    "He is a skilled wizard and sorcerer.",
    "He fought against Lord Voldemort and his followers.",
    "He has a pet owl named Snowy."
  ]
}
"""

        self.assertIsInstance(response, Character)
        assert response.name.startswith('Harry')
        assert response.name.endswith('Potter')
    def test_get_strutured_output_ollama(self):
        slm = StructuredLangModel('llama3.2')
        response = slm.get_response('Tell me about Harry Potter', '', response_model=Character)
        expected = """
{
  "name": "Harry James Potter",
  "age": 37,
  "fact": [
    "He is the chosen one.",
    "He has a lightning-shaped scar on his forehead.",
    "He is the son of James and Lily Potter.",
    "He attended Hogwarts School of Witchcraft and Wizardry.",
    "He is a skilled wizard and sorcerer.",
    "He fought against Lord Voldemort and his followers.",
    "He has a pet owl named Snowy."
  ]
}
"""

        self.assertIsInstance(response, Character)
        assert response.name.startswith('Harry')
        assert response.name.endswith('Potter')

if __name__ == '__main__':
    unittest.main()
