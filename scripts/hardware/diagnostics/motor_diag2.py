#!/usr/bin/env python3
"""
Thorough diagnostic: try every pin/polarity combination.
Physical pin 11 = GPIO 17, Physical pin 13 = GPIO 27.
"""

import lgpio
import time

GPIO17 = 17  # physical pin 11
GPIO27 = 27  # physical pin 13

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, GPIO17, 1)
lgpio.gpio_claim_output(h, GPIO27, 1)

def do_pulses(pul_pin, dir_pin, pul_active, dir_active, n=100, delay=0.003):
    """Send pulses with specified polarity."""
    pul_inactive = 1 - pul_active
    # Set direction
    lgpio.gpio_write(h, dir_pin, dir_active)
    time.sleep(0.01)
    for _ in range(n):
        lgpio.gpio_write(h, pul_pin, pul_active)
        time.sleep(delay)
        lgpio.gpio_write(h, pul_pin, pul_inactive)
        time.sleep(delay)
    # Reset both
    lgpio.gpio_write(h, pul_pin, 1)
    lgpio.gpio_write(h, dir_pin, 1)

print("=== Thorough Motor Diagnostic ===")
print("Listen/watch for ANY movement, vibration, or clicking.\n")

# Verify GPIO toggling
print("[Check] Verifying GPIO pins respond...")
for pin, name in [(GPIO17, "GPIO17/pin11"), (GPIO27, "GPIO27/pin13")]:
    lgpio.gpio_write(h, pin, 0)
    time.sleep(0.01)
    lgpio.gpio_write(h, pin, 1)
    time.sleep(0.01)
    print(f"  {name}: toggling OK")
print()

tests = [
    ("PUL=GPIO17 DIR=GPIO27 active-LOW",  GPIO17, GPIO27, 0, 0),
    ("PUL=GPIO17 DIR=GPIO27 active-HIGH", GPIO17, GPIO27, 1, 1),
    ("PUL=GPIO27 DIR=GPIO17 active-LOW",  GPIO27, GPIO17, 0, 0),
    ("PUL=GPIO27 DIR=GPIO17 active-HIGH", GPIO27, GPIO17, 1, 1),
    ("PUL=GPIO17 DIR=GPIO27 pul-LOW dir-HIGH",  GPIO17, GPIO27, 0, 1),
    ("PUL=GPIO17 DIR=GPIO27 pul-HIGH dir-LOW",  GPIO17, GPIO27, 1, 0),
    ("PUL=GPIO27 DIR=GPIO17 pul-LOW dir-HIGH",  GPIO27, GPIO17, 0, 1),
    ("PUL=GPIO27 DIR=GPIO17 pul-HIGH dir-LOW",  GPIO27, GPIO17, 1, 0),
]

for i, (desc, pul, dir_, pul_act, dir_act) in enumerate(tests, 1):
    print(f"[Test {i}/8] {desc}...")
    time.sleep(0.5)
    do_pulses(pul, dir_, pul_act, dir_act, n=100, delay=0.003)
    time.sleep(1.5)

print("\n[Test 9/9] VERY slow toggle - both pins alternating (listen for clicks)...")
time.sleep(0.5)
for _ in range(20):
    lgpio.gpio_write(h, GPIO17, 0)
    lgpio.gpio_write(h, GPIO27, 0)
    time.sleep(0.1)
    lgpio.gpio_write(h, GPIO17, 1)
    lgpio.gpio_write(h, GPIO27, 1)
    time.sleep(0.1)

print("\n=== Done ===")
print("If NOTHING moved in any test, the issue is wiring/connection.")
print("Check that wires are firmly seated in the GPIO header pins")
print("and that the driver's PUL/DIR terminals have solid connections.")

for pin in [GPIO17, GPIO27]:
    lgpio.gpio_write(h, pin, 1)
lgpio.gpiochip_close(h)
