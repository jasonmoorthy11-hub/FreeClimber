"""NEMA 34 motor control via 2HSS86 driver for RING assay.

Supports GPIO on Raspberry Pi (lgpio) with mock mode for development.
Default pins: Physical 11 (GPIO17) = PUL, Physical 13 (GPIO27) = DIR

CRITICAL: Physical pin numbers != GPIO/BCM numbers!
- Physical 11 = GPIO 17 (pulse)
- Physical 13 = GPIO 27 (direction)
- Physical 6 = GND
"""

import atexit
import logging
import platform
import time

logger = logging.getLogger(__name__)

# Default GPIO pins (BCM numbering)
DEFAULT_PUL_PIN = 17  # Physical pin 11
DEFAULT_DIR_PIN = 27  # Physical pin 13

# DIP switch presets: microstep setting -> steps per revolution
MICROSTEP_TABLE = {
    400: 400,
    800: 800,
    1600: 1600,
    3200: 3200,
    6400: 6400,
    12800: 12800,
    25600: 25600,
    51200: 51200,
}


class RINGMotor:
    """Controls NEMA 34 stepper motor via 2HSS86 driver.

    Uses lgpio on Pi, mock mode on non-Pi systems.
    """

    def __init__(self, pul_pin: int = DEFAULT_PUL_PIN, dir_pin: int = DEFAULT_DIR_PIN,
                 steps_per_rev: int = 6400, mock: bool = None):
        self.pul_pin = pul_pin
        self.dir_pin = dir_pin
        self.steps_per_rev = steps_per_rev
        self.current_position = 0  # steps from home

        # Auto-detect mock mode
        if mock is None:
            self.mock = not self._is_raspberry_pi()
        else:
            self.mock = mock

        if self.mock:
            logger.info("Motor in MOCK mode (not on Pi)")
            self._h = None
        else:
            self._init_gpio()

        atexit.register(self.cleanup)

    def _is_raspberry_pi(self) -> bool:
        if platform.system() != 'Linux':
            return False
        try:
            with open('/proc/device-tree/model') as f:
                return 'raspberry pi' in f.read().lower()
        except FileNotFoundError:
            return False

    def _init_gpio(self):
        try:
            import lgpio
            self._h = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._h, self.pul_pin)
            lgpio.gpio_claim_output(self._h, self.dir_pin)
            logger.info(f"GPIO initialized: PUL={self.pul_pin}, DIR={self.dir_pin}")
        except Exception as e:
            logger.error(f"GPIO init failed: {e}. Switching to mock mode.")
            self.mock = True
            self._h = None

    def _pulse(self, delay: float = 0.0001):
        """Send one step pulse."""
        if self.mock:
            return
        import lgpio
        lgpio.gpio_write(self._h, self.pul_pin, 1)
        time.sleep(delay)
        lgpio.gpio_write(self._h, self.pul_pin, 0)
        time.sleep(delay)

    def _set_direction(self, clockwise: bool = True):
        """Set rotation direction."""
        if self.mock:
            return
        import lgpio
        lgpio.gpio_write(self._h, self.dir_pin, 1 if clockwise else 0)
        time.sleep(0.001)

    def rotate(self, degrees: float, direction: str = 'cw', speed: float = None):
        """Rotate motor by specified degrees.

        Args:
            degrees: rotation angle
            direction: 'cw' or 'ccw'
            speed: delay between pulses (lower = faster). Default auto-calculated.
        """
        steps = int(self.steps_per_rev * degrees / 360)
        clockwise = direction.lower() == 'cw'

        if speed is None:
            speed = 0.0002  # default moderate speed

        logger.info(f"Rotating {degrees}deg {'CW' if clockwise else 'CCW'} ({steps} steps)")

        self._set_direction(clockwise)

        # Speed ramping: accelerate for first 10%, decelerate for last 10%
        ramp_steps = max(steps // 10, 1)

        for i in range(steps):
            if i < ramp_steps:
                # Accelerate
                delay = speed * (3 - 2 * i / ramp_steps)
            elif i > steps - ramp_steps:
                # Decelerate
                remaining = steps - i
                delay = speed * (3 - 2 * remaining / ramp_steps)
            else:
                delay = speed

            self._pulse(delay)

        self.current_position += steps if clockwise else -steps
        logger.info(f"Rotation complete. Position: {self.current_position} steps")

    def flip_180(self, speed: float = None):
        """Rotate 180 degrees for vial inversion."""
        self.rotate(180, direction='cw', speed=speed)

    def tap_sequence(self, n_taps: int = 3, interval: float = 0.5):
        """Standard RING tapping protocol.

        Per JoVE protocol: rapid taps to dislodge flies to bottom.
        """
        logger.info(f"Tap sequence: {n_taps} taps, {interval}s interval")

        for i in range(n_taps):
            # Quick tap: small rotation forward then back
            self.rotate(5, direction='cw', speed=0.00005)
            self.rotate(5, direction='ccw', speed=0.00005)
            if i < n_taps - 1:
                time.sleep(interval)

        logger.info("Tap sequence complete")

    def home(self):
        """Return to home position (0 steps)."""
        if self.current_position == 0:
            return
        direction = 'ccw' if self.current_position > 0 else 'cw'
        degrees = abs(self.current_position) * 360 / self.steps_per_rev
        self.rotate(degrees, direction=direction)
        self.current_position = 0

    def cleanup(self):
        """Release GPIO resources."""
        if not self.mock and self._h is not None:
            try:
                import lgpio
                lgpio.gpiochip_close(self._h)
                logger.info("GPIO cleaned up")
            except Exception:
                pass
            self._h = None
