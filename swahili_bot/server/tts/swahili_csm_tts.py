"""
Swahili CSM-1B TTS Server

This module provides a Modal server for the Swahili Conversational Speech Model (CSM-1B),
fine-tuned for Swahili text-to-speech with configurable speaker IDs.
"""

import asyncio
import json
import threading
import modal

# Regional configuration
SERVICE_REGIONS = ["us-west-1", "us-sanjose-1", "westus"]

app = modal.App("swahili-csm-tts")

# Create image with Swahili CSM model
swahili_tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "fastapi[standard]",
        "uvicorn[standard]",
    )
)

SAMPLE_RATE = 24000
UVICORN_PORT = 8000

with swahili_tts_image.imports():
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    from fastapi import FastAPI, WebSocket
    import uvicorn


@app.cls(
    image=swahili_tts_image,
    gpu="A10G",
    enable_memory_snapshot=True,
    region=SERVICE_REGIONS,
    scaledown_window=10,
)
class SwahiliTTS:
    """Swahili TTS service using CSM-1B model."""

    @modal.enter(snap=True)
    def load_model(self):
        """Load the Swahili CSM-1B model."""
        self.tunnel_ctx = None
        self.tunnel = None
        self.websocket_url = None

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

    @modal.enter(snap=False)
    def _start_server(self):
        """Start the WebSocket server."""
        self.web_app = FastAPI()

        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            await ws.accept()

            async def recv_loop(ws):
                while True:
                    msg = await ws.receive_text()
                    try:
                        json_data = json.loads(msg)
                        if json_data.get("type") == "prompt":
                            text = json_data.get("text", "")
                            speaker_id = json_data.get("speaker_id", 22)
                            max_tokens = json_data.get("max_tokens", 250)

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

                            # Send raw audio bytes
                            await ws.send_bytes(audio_bytes)

                    except Exception as e:
                        print(f"Error in TTS synthesis: {e}")
                        continue

            try:
                await recv_loop(ws)
            except Exception as e:
                print(f"WebSocket error: {e}")

        def start_server():
            uvicorn.run(self.web_app, host="0.0.0.0", port=UVICORN_PORT)

        self.server_thread = threading.Thread(target=start_server, daemon=True)
        self.server_thread.start()

        self.tunnel_ctx = modal.forward(UVICORN_PORT)
        self.tunnel = self.tunnel_ctx.__enter__()
        self.websocket_url = self.tunnel.url.replace("https://", "wss://") + "/ws"
        print(f"Swahili TTS Websocket URL: {self.websocket_url}")

    @modal.method()
    async def run_tunnel_client(self, d: modal.Dict):
        """Share the websocket URL via Modal Dict."""
        try:
            print(f"Sending Swahili TTS websocket url: {self.websocket_url}")
            await d.put.aio("url", self.websocket_url)

            while True:
                await asyncio.sleep(1.0)

        except Exception as e:
            print(f"Error running tunnel client: {type(e)}: {e}")


@app.local_entrypoint()
def main():
    """Test the TTS service."""
    print("Swahili TTS service ready!")
