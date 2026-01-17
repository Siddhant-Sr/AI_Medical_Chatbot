import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
from groq import Groq
from app.config import GROQ_API_KEY


def record_audio(
    output_path: str,
    duration: int = 5,
    sample_rate: int = 16000,
):
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    write(output_path, sample_rate, audio)


def transcribe_audio(audio_path: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            language="en",
        )

    return transcription.text
