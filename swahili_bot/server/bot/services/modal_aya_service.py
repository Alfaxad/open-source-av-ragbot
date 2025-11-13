from loguru import logger
import sys

from pipecat.frames.frames import StopFrame, CancelFrame
from pipecat.services.openai.llm import OpenAILLMService

from server.bot.services.modal_services import ModalTunnelManager

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass


class ModalAyaLLMService(OpenAILLMService):
    """Aya-101 LLM service using vLLM backend."""

    def __init__(
        self,
        *args,
        modal_tunnel_manager: ModalTunnelManager = None,
        base_url: str = None,
        **kwargs):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = "super-secret-key"

        self.modal_tunnel_manager = modal_tunnel_manager
        self.base_url = base_url
        if self.modal_tunnel_manager:
            logger.info(f"Using Modal Tunnels for Aya-101")
        if self.base_url:
            logger.info(f"Using Aya-101 URL: {self.base_url}")
        else:
            raise Exception("base_url must be provided for Aya-101")

        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        super().__init__(*args, base_url=base_url, **kwargs)

    async def stop(self, frame: StopFrame):
        await super().stop(frame)
        self._cleanup()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._cleanup()

    def _cleanup(self):
        if self.modal_tunnel_manager:
            self.modal_tunnel_manager.close()
