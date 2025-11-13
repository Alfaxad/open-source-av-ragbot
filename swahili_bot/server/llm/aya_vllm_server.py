"""
Aya-101 vLLM Server for Swahili Conversations

This module provides a Modal server running the Aya-101 multilingual model
optimized with vLLM for fast inference in Swahili conversations.
"""

import modal
import subprocess

app = modal.App("swahili-aya-llm")

MODEL_NAME = "CohereForAI/aya-101"

# Create image with vLLM
aya_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "vllm==0.8.2",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

@app.cls(
    image=aya_image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    enable_memory_snapshot=True,
    timeout=60 * 30,  # 30 minutes
)
class AyaLLM:
    """Aya-101 LLM service using vLLM for fast inference."""

    @modal.enter(snap=True)
    def start_server(self):
        """Start the vLLM server."""
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
            "--port", "8000",
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
                response = requests.get("http://localhost:8000/health")
                if response.status_code == 200:
                    print("vLLM server is ready!")
                    break
            except:
                pass
            time.sleep(1)
            if i == max_retries - 1:
                raise Exception("vLLM server failed to start")

    @modal.exit()
    def stop_server(self):
        """Stop the vLLM server."""
        if hasattr(self, 'process'):
            self.process.terminate()
            self.process.wait()

    @modal.web_server(port=8000)
    def serve(self):
        """Expose the vLLM server."""
        # vLLM server is already running on port 8000
        pass


@app.local_entrypoint()
def main():
    """Test the LLM service."""
    print("Aya-101 LLM service ready!")
