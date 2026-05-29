#!/usr/bin/env python3
"""Phase 2A: Direct GPIO source drive (common-cathode).

Tests if Pi GPIO can SOURCE enough current through the opto-LED.
No transistor, no breadboard — just GPIO → resistor → PUL+, GND → PUL-.

WIRING (disconnect ALL breadboard wires first):
  Pi Pin 6  (GND)    → PUL-  (screw terminal)
  Pi Pin 6  (GND)    → DIR-  (screw terminal)
  Pi Pin 2  (5V)     → DIR+  (screw terminal)
  GPIO 17 (Pin 11)   → 1K resistor → PUL+  (screw terminal)

WARNING: This may NOT work. GPIO sources only 3.3V.
  Current = (3.3V - 1.2V) / (1000 + 330) ≈ 1.6mA
  Driver needs 7-10mA minimum.
  If this fails, that's expected — proceed to Phase 2B.
"""

import sys
import time

try:
    import lgpio
except ImportError:
    print("ERROR: lgpio not available. Run this on the Pi.")
    sys.exit(1)

PUL = 17
DIR = 27

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, PUL, 0, lgpio.SET_PULL_NONE)
lgpio.gpio_claim_output(h, DIR, 0, lgpio.SET_PULL_NONE)

print("=" * 60)
print("PHASE 2A: Direct GPIO Source Drive (Common-Cathode)")
print("=" * 60)
print()
print("GPIO → 1K resistor → PUL+, GND → PUL-")
print("Logic: HIGH = pulse ON, LOW = pulse OFF")
print("Expected current: ~1.6mA (may be too low)")
print()

# Set direction
lgpio.gpio_write(h, DIR, 1)
time.sleep(0.01)

print("Sending 200 pulses at 3ms period (active-HIGH)...")
time.sleep(1)

for i in range(200):
    lgpio.gpio_write(h, PUL, 1)  # pulse ON
    time.sleep(0.0015)
    lgpio.gpio_write(h, PUL, 0)  # pulse OFF
    time.sleep(0.0015)

print("Batch 1 done. Trying slower pulses (10ms)...")
time.sleep(1)

for i in range(100):
    lgpio.gpio_write(h, PUL, 1)
    time.sleep(0.005)
    lgpio.gpio_write(h, PUL, 0)
    time.sleep(0.005)

print()
print("Did the motor move?")
print("  YES → GPIO can source enough current. Use this wiring.")
print("  NO  → Expected. Current too low. Proceed to Phase 2B (sink mode).")

lgpio.gpiochip_close(h)
