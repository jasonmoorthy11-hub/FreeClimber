"""NEMA 34 motor control via 2HSS86 driver for RING assay.

Supports GPIO on Raspberry Pi (lgpio) with mock mode for development.
Default pins: Physical 11 (GPIO17) = PUL, Physical 13 (GPIO27) = DIR

Supports two wiring modes:
  - "sink" (default): Common-anode. 5V → PUL+, GPIO → PUL-.
    Inverted logic: GPIO LOW = pulse ON, HIGH = pulse OFF.
    Pi sinks ~10mA from 5V — sufficient for 2HSS86 opto-input.
  - "source": Common-cathode. GPIO → resistor → PUL+, GND → PUL-.
    Normal logic: GPIO HIGH = pulse ON, LOW = pulse OFF.
    Requires transistor circuit — GPIO alone can't source enough current.

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

DEFAULT_PUL_PIN = 17  # Physical pin 11
DEFAULT_DIR_PIN = 27  # Physical pin 13

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
    """Controls NEMA 34 stepper motor via 2HSS86 driver."""

    def __init__(self, pul_pin: int = DEFAULT_PUL_PIN, dir_pin: int = DEFAULT_DIR_PIN,
                 steps_per_rev: int = 400, mock: bool = None,
                 wiring_mode: str = "sink"):
        self.pul_pin = pul_pin
        self.dir_pin = dir_pin
        self.steps_per_rev = steps_per_rev
        self.current_position = 0
        self.wiring_mode = wiring_mode  # "sink" (inverted) or "source" (normal)

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
            self._lgpio = lgpio
            self._h = lgpio.gpiochip_open(0)

            # Idle state depends on wiring mode:
            # sink mode: idle HIGH (no current through opto)
            # source mode: idle LOW (no current through opto)
            idle = 1 if self.wiring_mode == "sink" else 0

            lgpio.gpio_claim_output(self._h, self.pul_pin, idle, lgpio.SET_PULL_NONE)
            lgpio.gpio_claim_output(self._h, self.dir_pin, idle, lgpio.SET_PULL_NONE)
            logger.info(f"GPIO initialized: PUL={self.pul_pin}, DIR={self.dir_pin}, "
                        f"mode={self.wiring_mode}, idle={'HIGH' if idle else 'LOW'}")
        except Exception as e:
            logger.error(f"GPIO init failed: {e}. Switching to mock mode.")
            self.mock = True
            self._h = None

    def _pulse(self, delay: float = 0.0005):
        if self.mock:
            return
        if self.wiring_mode == "sink":
            # Inverted: LOW = active (current flows 5V → opto → GPIO sink)
            self._lgpio.gpio_write(self._h, self.pul_pin, 0)
            time.sleep(delay)
            self._lgpio.gpio_write(self._h, self.pul_pin, 1)
            time.sleep(delay)
        else:
            # Normal: HIGH = active
            self._lgpio.gpio_write(self._h, self.pul_pin, 1)
            time.sleep(delay)
            self._lgpio.gpio_write(self._h, self.pul_pin, 0)
            time.sleep(delay)

    def _set_direction(self, clockwise: bool = True):
        if self.mock:
            return
        if self.wiring_mode == "sink":
            # Inverted: LOW = active
            self._lgpio.gpio_write(self._h, self.dir_pin, 0 if clockwise else 1)
        else:
            self._lgpio.gpio_write(self._h, self.dir_pin, 1 if clockwise else 0)
        time.sleep(0.001)

    def diagnose(self) -> dict:
        """Run startup diagnostic — returns dict of check results."""
        results = {}
        results['mock'] = self.mock
        results['wiring_mode'] = self.wiring_mode
        results['steps_per_rev'] = self.steps_per_rev

        if self.mock:
            results['gpio'] = 'skipped (mock mode)'
            return results

        try:
            idle = 1 if self.wiring_mode == "sink" else 0
            active = 0 if self.wiring_mode == "sink" else 1

            # Toggle PUL pin and verify
            self._lgpio.gpio_write(self._h, self.pul_pin, active)
            time.sleep(0.01)
            self._lgpio.gpio_write(self._h, self.pul_pin, idle)
            results['pul_toggle'] = 'ok'

            # Toggle DIR pin and verify
            self._lgpio.gpio_write(self._h, self.dir_pin, active)
            time.sleep(0.01)
            self._lgpio.gpio_write(self._h, self.dir_pin, idle)
            results['dir_toggle'] = 'ok'

            # Send 10 test pulses (should be invisible at 400 steps/rev = 9°)
            for _ in range(10):
                self._pulse(0.001)
            results['test_pulses'] = 'sent 10 pulses'
            results['gpio'] = 'ok'
        except Exception as e:
            results['gpio'] = f'error: {e}'

        return results

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
            speed = 0.0005

        logger.info(f"Rotating {degrees}deg {'CW' if clockwise else 'CCW'} ({steps} steps)")

        self._set_direction(clockwise)

        ramp_steps = max(steps // 10, 1)

        for i in range(steps):
            if i < ramp_steps:
                delay = speed * (3 - 2 * i / ramp_steps)
            elif i > steps - ramp_steps:
                remaining = steps - i
                delay = speed * (3 - 2 * remaining / ramp_steps)
            else:
                delay = speed

            self._pulse(delay)

        self.current_position += steps if clockwise else -steps
        logger.info(f"Rotation complete. Position: {self.current_position} steps")

    def flip_180(self, speed: float = None):
        self.rotate(180, direction='cw', speed=speed)

    def tap_sequence(self, n_taps: int = 3, interval: float = 0.5):
        """Standard RING tapping protocol."""
        logger.info(f"Tap sequence: {n_taps} taps, {interval}s interval")

        for i in range(n_taps):
            self.rotate(5, direction='cw', speed=0.00015)
            self.rotate(5, direction='ccw', speed=0.00015)
            if i < n_taps - 1:
                time.sleep(interval)

        logger.info("Tap sequence complete")

    def home(self):
        if self.current_position == 0:
            return
        direction = 'ccw' if self.current_position > 0 else 'cw'
        degrees = abs(self.current_position) * 360 / self.steps_per_rev
        self.rotate(degrees, direction=direction)
        self.current_position = 0

    def cleanup(self):
        if not self.mock and self._h is not None:
            try:
                # Return pins to idle state before closing
                idle = 1 if self.wiring_mode == "sink" else 0
                self._lgpio.gpio_write(self._h, self.pul_pin, idle)
                self._lgpio.gpio_write(self._h, self.dir_pin, idle)
                self._lgpio.gpiochip_close(self._h)
                logger.info("GPIO cleaned up")
            except Exception:
                pass
            self._h = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("Motor self-test (sink mode, 400 steps/rev)")
    motor = RINGMotor(wiring_mode="sink", steps_per_rev=400)

    print("\nRunning diagnostics...")
    diag = motor.diagnose()
    for k, v in diag.items():
        print(f"  {k}: {v}")

    if not motor.mock:
        print("\nRotating 90° CW...")
        motor.rotate(90, 'cw')
        time.sleep(1)
        print("Rotating 90° CCW...")
        motor.rotate(90, 'ccw')
        print("\nDone. Motor should be back at start position.")

    motor.cleanup()
