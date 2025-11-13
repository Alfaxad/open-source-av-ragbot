"""
Swahili CSM-1B TTS Server

This module provides a Modal server for the Swahili Conversational Speech Model (CSM-1B),
fine-tuned for Swahili text-to-speech with configurable speaker IDs.
"""

import modal
import torch
import json
import base64
import numpy as np

app = modal.App("swahili-csm-tts")

# Create image with Swahili CSM model
swahili_tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "websockets",
    )
)

@app.cls(
    image=swahili_tts_image,
    gpu=modal.gpu.A10G(),
    enable_memory_snapshot=True,
)
class SwahiliTTS:
    """Swahili TTS service using CSM-1B model."""

    @modal.enter(snap=True)
    def load_model(self):
        """Load the Swahili CSM-1B model."""
        from transformers import AutoTokenizer, AutoModel

        print("Loading Swahili CSM-1B model...")
        model_name = "Nadhari/swa-csm-1b"

        self.processor = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        self.model.eval()
        print("Swahili CSM-1B model loaded successfully")

    @modal.web_endpoint(method="POST")
    async def synthesize(self, request: dict):
        """
        HTTP endpoint for TTS synthesis.

        Expects JSON with:
        - text: Text to synthesize
        - speaker_id: Speaker ID (default: 22)
        - max_tokens: Maximum tokens to generate (default: 250)
        """
        try:
            text = request.get("text", "")
            speaker_id = request.get("speaker_id", 22)
            max_tokens = request.get("max_tokens", 250)

            # Prepare conversation format
            conversation = [
                {
                    "role": str(speaker_id),
                    "content": [{"type": "text", "text": text}],
                },
            ]

            # Generate audio
            with torch.no_grad():
                audio_values = self.model.generate(
                    **self.processor.apply_chat_template(
                        conversation,
                        tokenize=True,
                        return_dict=True,
                    ).to("cuda"),
                    max_new_tokens=max_tokens,
                    output_audio=True,
                )

            # Convert to numpy and encode
            audio_array = audio_values[0].cpu().numpy()
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "audio": audio_b64,
                "sample_rate": 24000,
            }

        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return {"error": str(e)}

    @modal.method()
    async def synthesize_websocket(self, ws_url: str):
        """
        WebSocket endpoint for real-time TTS synthesis.
        """
        import websockets

        async with websockets.connect(ws_url) as websocket:
            print("TTS WebSocket connection established")

            try:
                async for message in websocket:
                    data = json.loads(message)

                    if data["type"] == "prompt":
                        text = data.get("text", "")
                        speaker_id = data.get("speaker_id", 22)
                        max_tokens = data.get("max_tokens", 250)

                        # Prepare conversation format
                        conversation = [
                            {
                                "role": str(speaker_id),
                                "content": [{"type": "text", "text": text}],
                            },
                        ]

                        # Generate audio
                        with torch.no_grad():
                            audio_values = self.model.generate(
                                **self.processor.apply_chat_template(
                                    conversation,
                                    tokenize=True,
                                    return_dict=True,
                                ).to("cuda"),
                                max_new_tokens=max_tokens,
                                output_audio=True,
                            )

                        # Convert to raw audio bytes and send
                        audio_array = audio_values[0].cpu().numpy()
                        audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()

                        # Send raw audio bytes (not base64)
                        await websocket.send(audio_bytes)

            except Exception as e:
                print(f"TTS WebSocket error: {e}")
                await websocket.send(json.dumps({"error": str(e)}))


@app.local_entrypoint()
def main():
    """Test the TTS service."""
    tts = SwahiliTTS()
    print("Swahili TTS service ready!")
