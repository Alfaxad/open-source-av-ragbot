import traceback
from typing import AsyncGenerator
from loguru import logger
import json
import sys

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    InterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from server.bot.services.modal_services import ModalWebsocketTTSService

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass

class ModalSwahiliTTSService(ModalWebsocketTTSService):
    """Swahili CSM-1B TTS service."""

    def __init__(
        self,
        speaker_id: int = 22,
        max_tokens: int = 250,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._speaker_id = speaker_id
        self._max_tokens = max_tokens
        self._running = False

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """

        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._running = False

        await super().push_frame(frame, direction)

    async def _receive_messages(self):
        """Receive and process messages from WebSocket.
        """
        async for message in self._get_websocket():
            try:
                await self.stop_ttfb_metrics()
                await self.push_frame(TTSAudioRawFrame(message, self.sample_rate, 1))
                logger.info(f"Received Swahili audio data of length {len(message)} bytes")
            except Exception as e:
                logger.error(f"Error decoding audio: {e}:{traceback.format_exc()}")
                await self.push_error(ErrorFrame(f"Error decoding audio: {e}"))

    async def run_tts(self, prompt: str) -> AsyncGenerator[Frame, None]:

        if not self._websocket:
            logger.error("Not connected to Swahili CSM TTS.")
            yield ErrorFrame("Not connected to Swahili CSM TTS.", fatal=True)
            return

        try:
            if not self._running:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._running = True

            tts_msg = {
                "type": "prompt",
                "text": prompt.strip(),
                "speaker_id": self._speaker_id,
                "max_tokens": self._max_tokens,
            }
            logger.info(f"Sending Swahili prompt: {tts_msg}")
            await self._websocket.send(json.dumps(tts_msg))
        except Exception as e:
            logger.error(f"Failed to send audio to Swahili CSM TTS: {e}")
            yield ErrorFrame(f"Failed to send audio to Swahili CSM TTS: {e}")

        yield None
