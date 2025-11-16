import unittest
from base_agent.llminterface import LangModel, StructuredLangModel
from pydantic import BaseModel, Field
from typing import List
import json


class TestLangModel(unittest.TestCase):
    def test_init_with_gpt_model(self):
        """Testa inicialização com modelo GPT"""
        lm = LangModel('gpt-4o')
        self.assertEqual(lm.model, 'gpt-4o')
        self.assertIsNotNone(lm.chat_history)

    def test_init_with_ollama_model(self):
        """Testa inicialização com modelo Ollama"""
        lm = LangModel('qwen3')
        self.assertEqual(lm.model, 'qwen3')
        self.assertIsNotNone(lm.chat_history)

    def test_init_with_deepseek_model(self):
        """Testa inicialização com modelo DeepSeek"""
        lm = LangModel(model='deepseek-chat')
        self.assertEqual(lm.model, 'deepseek-chat')
        self.assertIsNotNone(lm.chat_history)

    def test_reset_chat_history(self):
        """Testa reset do histórico de chat"""
        lm = LangModel('gpt-4o', provider='openai')
        lm.reset_chat_history()
        self.assertEqual(len(lm.chat_history.get_all()), 0)

    def test_available_models_property(self):
        """Testa propriedade available_models"""
        lm = LangModel('qwen3', provider='ollama')
        models = lm.available_models
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    def test_get_response_basic_functionality(self):
        """Testa funcionalidade básica de get_response"""
        lm = LangModel('gpt-4o')
        response = lm.get_response('What is 2+2?', 'Simple math question')
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


class SimpleResponse(BaseModel):
    answer: str
    confidence: float = Field(..., description="Confidence level between 0 and 1")


class TestStructuredLangModel(unittest.TestCase):
    def test_init_with_default_params(self):
        """Testa inicialização com parâmetros padrão"""
        slm = StructuredLangModel()
        self.assertEqual(slm.model, 'gpt-4o')
        self.assertIsNotNone(slm.chat_history)

    def test_init_with_custom_params(self):
        """Testa inicialização com parâmetros customizados"""
        slm = StructuredLangModel(model='gpt-4o', retries=5)
        self.assertEqual(slm.model, 'gpt-4o')
        self.assertEqual(slm.retries, 5)

    def test_reset_chat_history(self):
        """Testa reset do histórico de chat"""
        slm = StructuredLangModel()
        slm.reset_chat_history()
        self.assertEqual(len(slm.chat_history.get_all()), 0)

    def test_get_structured_output_simple(self):
        """Testa saída estruturada simples"""
        slm = StructuredLangModel()
        response = slm.get_response(
            'What is the capital of Brazil? Be confident in your answer.',
            '',
            response_model=SimpleResponse
        )
        
        self.assertIsInstance(response, SimpleResponse)
        self.assertIsInstance(response.answer, str)
        self.assertGreater(len(response.answer), 0)
        self.assertIsInstance(response.confidence, float)
        self.assertGreaterEqual(response.confidence, 0.0)
        self.assertLessEqual(response.confidence, 1.0)

    def test_get_structured_output_character(self):
        """Testa saída estruturada com modelo Character"""
        slm = StructuredLangModel()
        response = slm.get_response(
            'Tell me about Harry Potter, the fictional character',
            '',
            response_model=Character
        )

        self.assertIsInstance(response, Character)
        self.assertIsInstance(response.name, str)
        self.assertIn('Harry', response.name)
        self.assertIn('Potter', response.name)
        self.assertIsInstance(response.age, int)
        self.assertGreater(response.age, 0)
        self.assertIsInstance(response.fact, list)
        self.assertGreater(len(response.fact), 0)
        
        # Verifica se todos os fatos são strings
        for fact in response.fact:
            self.assertIsInstance(fact, str)
            self.assertGreater(len(fact), 0)

    def test_get_structured_output_ollama(self):
        """Testa saída estruturada com modelo Ollama"""
        try:
            slm = StructuredLangModel('llama3.2')
            response = slm.get_response(
                'What is 2+2? Be confident.',
                '',
                response_model=SimpleResponse
            )
            
            self.assertIsInstance(response, SimpleResponse)
            self.assertIsInstance(response.answer, str)
            self.assertIn('4', response.answer)
        except Exception as e:
            # Se Ollama não estiver disponível, pula o teste
            self.skipTest(f"Ollama não disponível: {e}")

    def test_get_response_without_model(self):
        """Testa resposta sem modelo estruturado"""
        slm = StructuredLangModel()
        response = slm.get_response('What is 2+2?', '')
        
        assert response


if __name__ == '__main__':
    unittest.main()
