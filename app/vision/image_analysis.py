import base64
from huggingface_hub import InferenceClient
from app.config import HF_TOKEN


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_medical_image(image_path: str) -> str:
    """
    Converts an image into a neutral textual description.
    No diagnosis, no advice.
    """

    client = InferenceClient(api_key=HF_TOKEN)
    encoded_image = image_to_base64(image_path)

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe the visible medical features in this image "
                            "in a neutral, factual way. Do not diagnose or suggest treatment."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content
