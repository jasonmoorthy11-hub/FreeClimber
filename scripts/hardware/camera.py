"""HQ Camera control for RING assay via Raspberry Pi.

Uses picamera2 on Pi, mock mode (reads example video) on other systems.
"""

import os
import time
import logging
import platform

logger = logging.getLogger(__name__)


class RINGCamera:
    """Controls Raspberry Pi HQ Camera for RING assay recording.

    Auto-detects Pi; on Mac/other, uses mock mode.
    """

    def __init__(self, mock: bool = None):
        if mock is None:
            self.mock = not self._is_raspberry_pi()
        else:
            self.mock = mock

        self._camera = None

        if not self.mock:
            self._init_camera()
        else:
            logger.info("Camera in MOCK mode (not on Pi)")

    def _is_raspberry_pi(self) -> bool:
        if platform.system() != 'Linux':
            return False
        try:
            with open('/proc/device-tree/model', 'r') as f:
                return 'raspberry pi' in f.read().lower()
        except FileNotFoundError:
            return False

    def _init_camera(self):
        try:
            from picamera2 import Picamera2
            self._camera = Picamera2()
            logger.info("Camera initialized")
        except ImportError:
            logger.warning("picamera2 not installed; switching to mock mode")
            self.mock = True
        except Exception as e:
            logger.error(f"Camera init failed: {e}. Switching to mock mode.")
            self.mock = True

    def record(self, duration: float, output_path: str,
               resolution: tuple = (1920, 1080), fps: int = 30):
        """Record video.

        Args:
            duration: recording time in seconds
            output_path: path for output video file (.mp4)
            resolution: (width, height)
            fps: frames per second
        """
        logger.info(f"Recording {duration}s to {output_path} ({resolution[0]}x{resolution[1]} @ {fps}fps)")

        if self.mock:
            logger.info(f"[MOCK] Would record {duration}s video to {output_path}")
            return

        from picamera2.encoders import H264Encoder
        from picamera2.outputs import FfmpegOutput

        config = self._camera.create_video_configuration(
            main={"size": resolution, "format": "RGB888"},
            controls={"FrameRate": fps}
        )
        self._camera.configure(config)

        encoder = H264Encoder()
        output = FfmpegOutput(output_path)

        self._camera.start_recording(encoder, output)
        time.sleep(duration)
        self._camera.stop_recording()

        logger.info(f"Recording complete: {output_path}")

    def capture_still(self, output_path: str, resolution: tuple = (4056, 3040)):
        """Capture a single still image (for calibration)."""
        logger.info(f"Capturing still to {output_path}")

        if self.mock:
            logger.info(f"[MOCK] Would capture still to {output_path}")
            return

        config = self._camera.create_still_configuration(
            main={"size": resolution}
        )
        self._camera.configure(config)
        self._camera.start()
        self._camera.capture_file(output_path)
        self._camera.stop()

        logger.info(f"Still captured: {output_path}")

    def preview(self):
        """Generator yielding JPEG frames for live preview (MJPEG stream)."""
        if self.mock:
            # Yield a placeholder frame
            import numpy as np
            import cv2
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "MOCK CAMERA", (150, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            _, jpeg = cv2.imencode('.jpg', placeholder)
            while True:
                yield jpeg.tobytes()
                time.sleep(1.0 / 15)
            return

        import io
        config = self._camera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self._camera.configure(config)
        self._camera.start()

        try:
            while True:
                frame = self._camera.capture_array()
                import cv2
                _, jpeg = cv2.imencode('.jpg', frame)
                yield jpeg.tobytes()
                time.sleep(1.0 / 15)
        finally:
            self._camera.stop()

    def get_info(self) -> dict:
        """Get camera information."""
        if self.mock:
            return {
                'model': 'Mock Camera',
                'resolution': (1920, 1080),
                'available': False,
            }

        try:
            props = self._camera.camera_properties
            return {
                'model': props.get('Model', 'Unknown'),
                'resolution': props.get('PixelArraySize', (0, 0)),
                'available': True,
            }
        except Exception:
            return {'model': 'Unknown', 'resolution': (0, 0), 'available': True}

    def close(self):
        """Release camera resources."""
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
