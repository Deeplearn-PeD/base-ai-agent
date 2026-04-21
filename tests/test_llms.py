import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from base_agent.llminterface import LangModel, StructuredLangModel
from pydantic import BaseModel, Field
from typing import List


class TestLangModel(unittest.TestCase):
    @patch.object(LangModel, "_get_model_instance")
    @patch.object(LangModel, "_fetch_provider_models")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_init_with_gpt_model(self, mock_fetch_models, mock_get_model_instance):
        mock_fetch_models.return_value = ["gpt-4-turbo", "gpt-4o"]
        mock_get_model_instance.return_value = MagicMock()
        mock_agent = MagicMock()
        with patch("base_agent.llminterface.Agent", return_value=mock_agent):
            LangModel("gpt-4-turbo", provider="openai")
            mock_get_model_instance.assert_called()

    @patch.object(LangModel, "_get_model_instance")
    @patch.object(LangModel, "_fetch_provider_models")
    def test_init_with_ollama_model(self, mock_fetch_models, mock_get_model_instance):
        mock_fetch_models.return_value = ["llama3", "llama3.1"]
        mock_get_model_instance.return_value = MagicMock()
        mock_agent = MagicMock()
        with patch("base_agent.llminterface.Agent", return_value=mock_agent):
            lm = LangModel("llama3", provider="ollama")
            self.assertEqual(lm.model, "llama3")

    @patch.object(LangModel, "_get_model_instance")
    @patch.object(LangModel, "_fetch_provider_models")
    def test_init_with_unsupported_model_falls_back(
        self, mock_fetch_models, mock_get_model_instance
    ):
        mock_fetch_models.return_value = ["llama3", "qwen3"]
        mock_get_model_instance.return_value = MagicMock()
        mock_agent = MagicMock()
        with patch("base_agent.llminterface.Agent", return_value=mock_agent):
            lm = LangModel("unsupported_model")
            self.assertNotEqual(lm.model, "unsupported_model")

    @patch.object(LangModel, "_get_model_instance")
    @patch.object(LangModel, "_fetch_provider_models")
    def test_get_response_with_gpt_model(
        self, mock_fetch_models, mock_get_model_instance
    ):
        mock_fetch_models.return_value = ["gpt-4o"]
        mock_get_model_instance.return_value = MagicMock()
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "test response"
        mock_result.new_messages = MagicMock(return_value=[])
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("base_agent.llminterface.Agent", return_value=mock_agent):
            lm = LangModel("gpt-4o", provider="openai")
            response = lm.get_response("question", "context")
            self.assertEqual(response, "test response")

    @patch.object(LangModel, "_get_model_instance")
    @patch.object(LangModel, "_fetch_provider_models")
    def test_get_response_with_ollama_model(
        self, mock_fetch_models, mock_get_model_instance
    ):
        mock_fetch_models.return_value = ["llama3"]
        mock_get_model_instance.return_value = MagicMock()
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "test response"
        mock_result.new_messages = MagicMock(return_value=[])
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("base_agent.llminterface.Agent", return_value=mock_agent):
            lm = LangModel("llama3", provider="ollama")
            response = lm.get_response("question", "context")
            self.assertEqual(response, "test response")


class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


class TestStructuredLangModel(unittest.TestCase):
    @patch.object(LangModel, "_get_model_instance")
    @patch.object(LangModel, "_fetch_provider_models")
    def test_get_structured_output(self, mock_fetch_models, mock_get_model_instance):
        mock_fetch_models.return_value = ["gpt-4o"]
        mock_get_model_instance.return_value = MagicMock()

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = Character(
            name="Harry Potter",
            age=37,
            fact=["He is the chosen one.", "He has a lightning-shaped scar."],
        )
        mock_result.new_messages = MagicMock(return_value=[])
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("base_agent.llminterface.Agent", return_value=mock_agent):
            slm = StructuredLangModel("gpt-4o")
            response = slm.get_response(
                "Tell me about Harry Potter", "", response_model=Character
            )
            self.assertIsInstance(response, Character)


if __name__ == "__main__":
    unittest.main()
