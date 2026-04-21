import yaml


class BasePersona:
    def __init__(self, name: str='Libby D. Bot', model: str='gpt-4o',  languages=['pt_BR','en'], ):
        self.name = name
        self.languages = languages
        self.active_language = languages[0]
        self.model = model
        self.context_prompt = None

    def set_language(self, language: str):
        if language in self.languages:
            self.active_language = language
        else:
            raise ValueError(f"Language {language} not supported by this persona.")