#!/usr/bin/env python3
"""
Motor control for RATTM MOTOR 2HSS86 hybrid servo driver.

Wiring (5V common anode + NPN transistor level shifting):
  PUL+ -> 5V (Pi physical pin 2)
  PUL- -> Collector of 2N2222 transistor 1
  DIR+ -> 5V (Pi physical pin 4)
  DIR- -> Collector of 2N2222 transistor 2
  Both emitters -> GND (Pi physical pin 6)
  Transistor 1 base -> 1K resistor -> Pi physical pin 11 (GPIO 17)
  Transistor 2 base -> 1K resistor -> Pi physical pin 13 (GPIO 27)

Logic (transistor inverts):
  GPIO HIGH -> transistor ON -> collector LOW -> 5V across opto -> pulse ON
  GPIO LOW  -> transistor OFF -> collector floats -> opto OFF

Motor: RATTM MOTOR 86HSE156-12N-BC38 (NEMA 34, 12Nm)
PSU: S-400-70 (70V 400W)
Driver: RATTM MOTOR 2HSS86
"""

import lgpio
import time
import sys

PUL = 17  # Physical pin 11
DIR = 27  # Physical pin 13

h = None


def setup():
    global h
    h = lgpio.gpiochip_open(0)
    for pin in [PUL, DIR]:
        lgpio.gpio_claim_output(h, pin, 0)  # Start LOW (inactive)
    print("GPIO initialized (transistor level-shifted, active-high).")


def cleanup():
    if h is not None:
        for pin in [PUL, DIR]:
            lgpio.gpio_write(h, pin, 0)
        lgpio.gpiochip_close(h)
    print("GPIO cleaned up.")


def set_direction(clockwise=True):
    if clockwise:
        lgpio.gpio_write(h, DIR, 1)
    else:
        lgpio.gpio_write(h, DIR, 0)


def pulse(delay_s):
    lgpio.gpio_write(h, PUL, 1)  # Transistor ON -> opto ON
    time.sleep(delay_s)
    lgpio.gpio_write(h, PUL, 0)  # Transistor OFF -> opto OFF
    time.sleep(delay_s)


def move(steps, delay_s=0.0005, clockwise=True):
    set_direction(clockwise)
    time.sleep(0.005)

    print(f"Moving {'CW' if clockwise else 'CCW'}: {steps} steps "
          f"@ {1/(2*delay_s):.0f} Hz")

    for i in range(steps):
        pulse(delay_s)

    print("Move complete.")


def test_motor():
    print("\n=== Motor Test ===")
    print("Starting in 2 seconds...")
    time.sleep(2)

    print("\n[1/3] 200 steps CW (slow)...")
    move(steps=200, delay_s=0.002, clockwise=True)
    time.sleep(1)

    print("[2/3] 200 steps CCW (slow)...")
    move(steps=200, delay_s=0.002, clockwise=False)
    time.sleep(1)

    print("[3/3] 1000 steps CW (faster)...")
    move(steps=1000, delay_s=0.0005, clockwise=True)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    try:
        setup()

        if len(sys.argv) > 1:
            cmd = sys.argv[1]
            if cmd == "test":
                test_motor()
            elif cmd == "cw":
                steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
                speed = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
                move(steps=steps, delay_s=speed, clockwise=True)
            elif cmd == "ccw":
                steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
                speed = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
                move(steps=steps, delay_s=speed, clockwise=False)
            else:
                print("Usage: motor_control.py [test|cw|ccw] [steps] [delay_s]")
                print("  test          - Run test sequence")
                print("  cw 400 0.001  - 400 steps clockwise, 1kHz")
                print("  ccw 400 0.001 - 400 steps counter-clockwise, 1kHz")
        else:
            test_motor()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cleanup()
