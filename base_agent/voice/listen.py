import whisper


class Listen:
    """
    The Listen class is responsible for transcribing audio files.

    Attributes:
        model: The model used for transcription.
        options: The options used for decoding the audio.
        language: The language of the audio. If not set, it will be detected from the audio.
    """
    def __init__(self):
        self.model = whisper.load_model("base")
        self.options = whisper.DecodingOptions()
        self.language = None

    def listen(self, audio_file: str)-> str:
        """
        Listen to an audio file and return the transcription
        :param audio_file: Audio file path
        :return: Transcription
        """
        audio = whisper.load_audio(audio_file)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        if self.language is None:
            _, probs = self.model.detect_language(mel)
            self.language = max(probs, key=probs.get)
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)
        return result.text
