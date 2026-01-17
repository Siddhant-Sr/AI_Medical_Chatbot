from app.audio.stt import record_audio, transcribe_audio
from app.audio.tts import speak_text
from app.vision.image_analysis import analyze_medical_image
from app.core.orchestrator import handle_user_request


def main():
    # -------- VOICE INPUT --------
    audio_path = "user_input.wav"
    record_audio(audio_path, duration=6)
    user_text = transcribe_audio(audio_path)

    # -------- OPTIONAL IMAGE --------
    image_path = "data/sample_image.jpg"  # set None if no image
    image_summary = None

    if image_path:
        image_summary = analyze_medical_image(image_path)

    # -------- ORCHESTRATION --------
    result = handle_user_request(
        user_text=user_text,
        image_summary=image_summary,
    )

    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES:\n", result["sources"])

    # -------- VOICE OUTPUT --------
    speak_text(result["answer"])


if __name__ == "__main__":
    main()
