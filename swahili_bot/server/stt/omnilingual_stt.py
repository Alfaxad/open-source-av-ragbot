"""
Omnilingual ASR Server for Swahili

This module provides a Modal server for the Omnilingual ASR model,
configured specifically for Swahili (swh_Latn) language transcription.
"""

import asyncio
import json
import base64
import threading
import modal

# Regional configuration
SERVICE_REGIONS = ["us-west-1", "us-sanjose-1", "westus"]

app = modal.App("swahili-omnilingual-transcription")

# Create image with Omnilingual ASR
omnilingual_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "omnilingual-asr",
        "numpy",
        "torch",
        "fastapi[standard]",
        "uvicorn[standard]",
    )
)

SAMPLE_RATE = 16000
UVICORN_PORT = 8000

with omnilingual_image.imports():
    import numpy as np
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from fastapi import FastAPI, WebSocket
    import uvicorn


@app.cls(
    image=omnilingual_image,
    gpu="A10G",
    enable_memory_snapshot=True,
    region=SERVICE_REGIONS,
    scaledown_window=10,
)
class SwahiliTranscriber:
    """Swahili transcription service using Omnilingual ASR CTC-1B."""

    @modal.enter(snap=True)
    def load_model(self):
        """Load the Omnilingual ASR model."""
        self.tunnel_ctx = None
        self.tunnel = None
        self.websocket_url = None

        print("Loading Omnilingual ASR CTC-1B for Swahili...")
        self.pipeline = ASRInferencePipeline(
            model_card="omniASR_CTC_1B",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("Omnilingual ASR model loaded successfully")

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
                        if json_data.get("type") == "set_vad":
                            # VAD control (ignored for now)
                            continue
                        elif json_data.get("type") == "audio":
                            # Decode audio
                            audio_b64 = json_data["audio"]
                            audio_bytes = base64.b64decode(audio_b64)
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                            # Transcribe
                            transcriptions = self.pipeline.transcribe(
                                [{"waveform": audio_array, "sample_rate": SAMPLE_RATE}],
                                lang=["swh_Latn"],  # Swahili
                                batch_size=1,
                            )

                            text = transcriptions[0]["text"] if transcriptions else ""
                            if text:
                                await ws.send_text(text)

                    except Exception as e:
                        print(f"Error in transcription: {e}")
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
        print(f"Swahili STT Websocket URL: {self.websocket_url}")

    @modal.method()
    async def run_tunnel_client(self, d: modal.Dict):
        """Share the websocket URL via Modal Dict."""
        try:
            print(f"Sending Swahili STT websocket url: {self.websocket_url}")
            await d.put.aio("url", self.websocket_url)

            while True:
                await asyncio.sleep(1.0)

        except Exception as e:
            print(f"Error running tunnel client: {type(e)}: {e}")


@app.local_entrypoint()
def main():
    """Test the transcription service."""
    print("Swahili Transcriber service ready!")
