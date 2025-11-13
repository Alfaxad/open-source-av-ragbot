"""
Aya-101 vLLM Server for Swahili Conversations

This module provides a Modal server running the Aya-101 multilingual model
optimized with vLLM for fast inference in Swahili conversations.
"""

import modal
import subprocess
import asyncio

from server import SERVICE_REGIONS

app = modal.App("swahili-aya-llm")

MODEL_NAME = "CohereForAI/aya-101"

# Create image with vLLM
aya_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "vllm==0.8.2",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

UVICORN_PORT = 8000

@app.cls(
    image=aya_image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    enable_memory_snapshot=True,
    region=SERVICE_REGIONS,
    scaledown_window=10,
    timeout=60 * 30,  # 30 minutes
)
class AyaLLM:
    """Aya-101 LLM service using vLLM for fast inference."""

    @modal.enter(snap=True)
    def start_server(self):
        """Start the vLLM server."""
        self.tunnel_ctx = None
        self.tunnel = None
        self.base_url = None

        print(f"Starting vLLM server for {MODEL_NAME}...")

        # Start vLLM server process
        cmd = [
            "vllm", "serve",
            MODEL_NAME,
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "auto",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--port", str(UVICORN_PORT),
            "--host", "0.0.0.0",
        ]

        print(f"Running command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)

        # Wait for server to be ready
        import time
        import requests

        max_retries = 60
        for i in range(max_retries):
            try:
                response = requests.get(f"http://localhost:{UVICORN_PORT}/health")
                if response.status_code == 200:
                    print("vLLM server is ready!")
                    break
            except:
                pass
            time.sleep(1)
            if i == max_retries - 1:
                raise Exception("vLLM server failed to start")

    @modal.enter(snap=False)
    def _start_tunnel(self):
        """Create tunnel to vLLM server."""
        self.tunnel_ctx = modal.forward(UVICORN_PORT)
        self.tunnel = self.tunnel_ctx.__enter__()
        self.base_url = self.tunnel.url
        print(f"Aya-101 LLM URL: {self.base_url}")

    @modal.exit()
    def stop_server(self):
        """Stop the vLLM server."""
        if hasattr(self, 'process'):
            self.process.terminate()
            self.process.wait()

    @modal.method()
    async def run_tunnel_client(self, d: modal.Dict):
        """Share the base URL via Modal Dict."""
        try:
            print(f"Sending Aya-101 LLM url: {self.base_url}")
            await d.put.aio("url", self.base_url)

            while True:
                await asyncio.sleep(1.0)

        except Exception as e:
            print(f"Error running tunnel client: {type(e)}: {e}")


@app.local_entrypoint()
def main():
    """Test the LLM service."""
    print("Aya-101 LLM service ready!")
