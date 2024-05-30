import unittest
from base_agent.voice.listen import Listen

class TestListen(unittest.TestCase):
    def setUp(self):
        self.listen = Listen()

    def test_init(self):
        self.assertIsNotNone(self.listen.model)
        self.assertIsNotNone(self.listen.options)
        self.assertIsNone(self.listen.language)

    def test_listen(self):
        # This test assumes that you have a valid audio file 'test_audio.wav' in your project directory
        result = self.listen.listen('voice/fixtures/audio_ptBR.mp3')
        expected = 'O sistema InfoDengue é um pipeline de coleta, harmonização e análise de dados semi-automatizado, '\
                    'que gera indicadores de situação epidemiológica da dengue e outras arboviroses a nível municipal.'

        self.assertIsInstance(result, expected)

if __name__ == '__main__':
    unittest.main()
