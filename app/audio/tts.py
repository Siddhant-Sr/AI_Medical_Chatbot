import os
import platform
import subprocess
from elevenlabs.client import ElevenLabs
import elevenlabs
from app.config import ELEVEN_API_KEY


def speak_text(text: str, output_path="response.mp3"):
    client = ElevenLabs(api_key=ELEVEN_API_KEY)

    audio = client.generate(
        text=text,
        voice="Aria",
        model="eleven_turbo_v2",
        output_format="mp3_22050_32",
    )

    elevenlabs.save(audio, output_path)

    system = platform.system()
    if system == "Windows":
        subprocess.run(
            [
                "powershell",
                "-c",
                f'(New-Object Media.SoundPlayer "{output_path}").PlaySync();',
            ]
        )
    elif system == "Darwin":
        subprocess.run(["afplay", output_path])
    elif system == "Linux":
        subprocess.run(["aplay", output_path])
