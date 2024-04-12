import pydantic
from base_agent.voice import talk
import yaml


class BasePersona:
    def __init__(self, name: str='Libby D. Bot', model: str='gpt-4-0125-preview',  languages=['pt_BR','en'], ):
        self.name = name
        self.languages = languages
        self.active_language = languages[0]
        self.model = model
        self.voice = talk.Speaker(language=self.active_language)
        self.say = self.voice.say
        self.context_prompt = base_prompt[self.active_language]

    def set_language(self, language: str):
        if language in self.languages:
            self.active_language = language
            self.voice = talk.Speaker(language=self.active_language)
            self.say = self.voice.say
            self.context_prompt = base_prompt[self.active_language]
        else:
            raise ValueError(f"Language {language} not supported by this persona.")