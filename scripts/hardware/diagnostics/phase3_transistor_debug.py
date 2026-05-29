#!/usr/bin/env python3
"""Phase 3: Transistor circuit debugging (if needed).

Only run this if Phase 2B (direct sink) works but you want the transistor
circuit for robustness/isolation.

Provides:
  - GPIO hold modes for multimeter probing (V_BE, V_CE)
  - LED blink indicator test
  - Pulse test through transistor circuit
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
print("PHASE 3: Transistor Circuit Debug")
print("=" * 60)
print()
print("This uses the ORIGINAL breadboard wiring (transistor circuit).")
print("Make sure breadboard is connected per WIRING_GUIDE_V2.")
print()

# --- Test 3A: Hold GPIO HIGH for multimeter probing ---
print("TEST 3A: Voltage measurement under load")
print("  GPIO 17 (PUL) will be held HIGH for 60 seconds.")
print("  Measure with multimeter:")
print("    V_BE: Red on base (row 5, col b), Black on emitter (row 5, col a)")
print("      Expected: ~0.7V (transistor conducting)")
print("      If ~0V: no base current (broken path)")
print("      If ~2.6V: emitter not grounded")
print("    V_CE: Red on collector (row 5, col c), Black on emitter (row 5, col a)")
print("      Expected: ~0.2V (saturated)")
print("      If ~5V: transistor is OFF")
print()

input("Press Enter to hold GPIO 17 HIGH for 60 seconds...")

lgpio.gpio_write(h, DIR, 1)
lgpio.gpio_write(h, PUL, 1)
print("GPIO 17 = HIGH. Probe now! (60 seconds)")

for remaining in range(60, 0, -10):
    print(f"  {remaining}s remaining...", flush=True)
    time.sleep(10)

lgpio.gpio_write(h, PUL, 0)
print("GPIO 17 back to LOW.")
print()

# --- Test 3B: Slow visible pulses ---
print("TEST 3B: Slow pulses through transistor circuit")
print("  Sending 50 pulses at 100ms period (very slow).")
print("  If you have an LED + 330 ohm across PUL+/PUL- on the driver,")
print("  it should blink visibly.")
print()

input("Press Enter to send slow pulses...")

lgpio.gpio_write(h, DIR, 1)
time.sleep(0.01)

for i in range(50):
    lgpio.gpio_write(h, PUL, 1)
    time.sleep(0.05)
    lgpio.gpio_write(h, PUL, 0)
    time.sleep(0.05)

print("Done. Did LED blink / motor move?")
print()

# --- Test 3C: Normal speed pulses ---
print("TEST 3C: Normal speed pulses (400 = 1 rev at 400 steps/rev)")

input("Press Enter to send 400 pulses...")

lgpio.gpio_write(h, DIR, 1)
time.sleep(0.01)

for i in range(400):
    lgpio.gpio_write(h, PUL, 1)
    time.sleep(0.0015)
    lgpio.gpio_write(h, PUL, 0)
    time.sleep(0.0015)

print("Done.")
print()
print("If nothing worked through transistor circuit but Phase 2B worked,")
print("skip the transistor — direct sink wiring is simpler and sufficient.")

lgpio.gpiochip_close(h)
