"""
Omnilingual ASR Server for Swahili

This module provides a Modal server for the Omnilingual ASR model,
configured specifically for Swahili (swh_Latn) language transcription.

The CTC-1B variant is used for fast real-time transcription (16-96x faster than real-time).
"""

import modal
import asyncio
import numpy as np
import torch
import json
import base64

app = modal.App("swahili-omnilingual-transcription")

# Create image with Omnilingual ASR
omnilingual_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "omnilingual-asr",  # This will install the correct torch version
        "numpy",
        "websockets",
        "fastapi[standard]",
    )
)

@app.cls(
    image=omnilingual_image,
    gpu=modal.gpu.A10G(),
    enable_memory_snapshot=True,
)
class SwahiliTranscriber:
    """Swahili transcription service using Omnilingual ASR CTC-1B."""

    @modal.enter(snap=True)
    def load_model(self):
        """Load the Omnilingual ASR model."""
        from omnilingual_asr import ASRInferencePipeline

        print("Loading Omnilingual ASR CTC-1B for Swahili...")
        self.pipeline = ASRInferencePipeline(
            model_card="omniASR_CTC_1B",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("Omnilingual ASR model loaded successfully")

    @modal.web_endpoint(method="POST")
    async def transcribe(self, audio_data: dict):
        """
        HTTP endpoint for transcription.

        Expects JSON with base64-encoded audio data.
        """
        try:
            # Decode audio from base64
            audio_bytes = base64.b64decode(audio_data["audio"])
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe
            transcriptions = self.pipeline.transcribe(
                [{"waveform": audio_array, "sample_rate": 16000}],
                lang=["swh_Latn"],  # Swahili language code
                batch_size=1,
            )

            text = transcriptions[0]["text"] if transcriptions else ""
            return {"text": text}

        except Exception as e:
            print(f"Transcription error: {e}")
            return {"error": str(e)}

    @modal.method()
    async def transcribe_websocket(self, ws_url: str):
        """
        WebSocket endpoint for real-time transcription.
        """
        import websockets

        async with websockets.connect(ws_url) as websocket:
            print("WebSocket connection established")

            try:
                async for message in websocket:
                    data = json.loads(message)

                    if data["type"] == "audio":
                        # Decode and transcribe audio
                        audio_bytes = base64.b64decode(data["audio"])
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                        transcriptions = self.pipeline.transcribe(
                            [{"waveform": audio_array, "sample_rate": 16000}],
                            lang=["swh_Latn"],
                            batch_size=1,
                        )

                        text = transcriptions[0]["text"] if transcriptions else ""

                        # Send transcription back
                        await websocket.send(text)

            except Exception as e:
                print(f"WebSocket error: {e}")
                await websocket.send(json.dumps({"error": str(e)}))


@app.local_entrypoint()
def main():
    """Test the transcription service."""
    transcriber = SwahiliTranscriber()
    print("Swahili Transcriber service ready!")
