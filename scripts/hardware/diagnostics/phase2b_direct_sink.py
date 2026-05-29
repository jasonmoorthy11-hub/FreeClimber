#!/usr/bin/env python3
"""Phase 2B: Direct GPIO sink drive (common-anode, INVERTED logic).

THE MOST LIKELY SOLUTION. Pi GPIO sinks current from 5V through opto-LED.
No transistor, no breadboard, no resistor needed (driver has internal 330 ohm).

WIRING (disconnect ALL breadboard wires first):
  Pi Pin 2  (5V)     → PUL+  (screw terminal)
  Pi Pin 2  (5V)     → DIR+  (screw terminal, or use Pin 4)
  GPIO 17 (Pin 11)   → PUL-  (screw terminal, DIRECT — no resistor)
  GPIO 27 (Pin 13)   → DIR-  (screw terminal, DIRECT — no resistor)

LOGIC IS INVERTED:
  GPIO LOW  = current flows from 5V through opto to GPIO = pulse ON
  GPIO HIGH = no current = pulse OFF

Current: (5V - 1.2V - 0.4V) / 330 ≈ 10.3mA — well within spec!

This is the simplest possible wiring. If this works, the transistor
circuit was never needed.
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
# Start HIGH (inactive for inverted logic)
lgpio.gpio_claim_output(h, PUL, 1, lgpio.SET_PULL_NONE)
lgpio.gpio_claim_output(h, DIR, 1, lgpio.SET_PULL_NONE)

print("=" * 60)
print("PHASE 2B: Direct GPIO Sink Drive (Common-Anode, Inverted)")
print("=" * 60)
print()
print("5V → PUL+, GPIO 17 → PUL- (direct)")
print("5V → DIR+, GPIO 27 → DIR- (direct)")
print("Logic: LOW = pulse ON, HIGH = pulse OFF (INVERTED)")
print("Expected current: ~10.3mA (sufficient!)")
print()

# Set direction: LOW = active (inverted)
lgpio.gpio_write(h, DIR, 0)
time.sleep(0.01)

# Test 1: slow pulses
print("Test 1: 50 slow pulses (20ms period) — watch for movement...")
time.sleep(1)

for i in range(50):
    lgpio.gpio_write(h, PUL, 0)  # pulse ON (inverted!)
    time.sleep(0.01)
    lgpio.gpio_write(h, PUL, 1)  # pulse OFF (inverted!)
    time.sleep(0.01)

time.sleep(1)

# Test 2: faster pulses
print("Test 2: 200 pulses (6ms period) — should see ~180° at 400 steps/rev...")
time.sleep(1)

for i in range(200):
    lgpio.gpio_write(h, PUL, 0)  # pulse ON
    time.sleep(0.003)
    lgpio.gpio_write(h, PUL, 1)  # pulse OFF
    time.sleep(0.003)

time.sleep(1)

# Test 3: full revolution
print("Test 3: 400 pulses (3ms period) — full revolution at 400 steps/rev...")
time.sleep(1)

for i in range(400):
    lgpio.gpio_write(h, PUL, 0)
    time.sleep(0.0015)
    lgpio.gpio_write(h, PUL, 1)
    time.sleep(0.0015)

# Return DIR to inactive
lgpio.gpio_write(h, DIR, 1)

print()
print("=" * 60)
print("RESULTS:")
print("  If motor moved → THIS IS THE SOLUTION!")
print("    The Pi's 3.3V GPIO couldn't SOURCE enough current,")
print("    but it CAN SINK ~10mA from 5V. No transistor needed.")
print("    Update motor.py to use inverted pulse logic.")
print()
print("  If motor did NOT move → problem is downstream.")
print("    Check: DIP switches, encoder cable, motor phase wiring.")
print("    Run Phase 1 manual tap test first.")
print("=" * 60)

lgpio.gpiochip_close(h)
