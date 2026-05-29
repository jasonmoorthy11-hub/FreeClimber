#!/usr/bin/env python3
"""Phase 1: Isolate the 2HSS86 driver and NEMA 34 motor.

Confirms driver/motor work INDEPENDENT of transistor circuit.
Run each test in order — stop at first failure.

BEFORE RUNNING:
  1. Set DIP switches on 2HSS86:
     SW1=ON, SW2=OFF, SW3=OFF  → 6.0A peak current
     SW4=OFF                   → full holding torque
     SW5=ON, SW6=ON, SW7=ON, SW8=ON → 400 pulses/rev
  2. Power cycle the 70V PSU after changing DIP switches
  3. Verify motor shaft is LOCKED (servo-locked = encoder working)
     If shaft spins freely → reseat encoder ribbon cable, power cycle

OPTO-LED TEST (do this with multimeter, power OFF, all wires disconnected):
  - Multimeter in DIODE mode (triangle-with-line symbol)
  - Red probe on PUL+, Black probe on PUL- → expect 1.0-1.2V
  - Swap probes → expect OL
  - Repeat for DIR+/DIR-
  - If OL both ways → opto-LED is dead, driver input burned out

MANUAL TAP TEST WIRING (disconnect ALL existing wires from PUL/DIR first):
  - Pi Pin 2 (5V) → PUL+ terminal (screw tight)
  - Pi Pin 2 (5V) → DIR+ terminal (or use Pin 4)
  - Pi Pin 6 (GND) → DIR- terminal (screw tight)
  - Loose GND wire from Pi Pin 6 — tap against PUL- terminal rapidly
  - At 400 pulses/rev: ~50 taps ≈ 45 degrees visible rotation
"""

import sys
import time

print("=" * 60)
print("PHASE 1: Driver & Motor Isolation Test")
print("=" * 60)

print("""
STEP 1A — DIP Switch Settings (do this first):
  SW1=ON  SW2=OFF SW3=OFF  → 6.0A peak
  SW4=OFF                  → full torque
  SW5=ON  SW6=ON  SW7=ON  SW8=ON → 400 pulses/rev
  Power cycle 70V PSU after changing switches!

STEP 1B — Servo Lock Test:
  With 70V on (green LED), try turning motor shaft by hand.
  Should feel STRONG resistance (locked).
  If it spins freely → encoder cable not seated.

STEP 1C — Opto-LED Diode Test (POWER OFF, all wires disconnected):
  Multimeter → diode mode
  Red on PUL+, Black on PUL- → expect 1.0-1.2V
  Swap → expect OL (open)
  Same for DIR+/DIR-
  Both OL = burned opto = driver needs replacement

STEP 1D — Manual Tap Test:
  Wire: Pi Pin 2 (5V) → PUL+
  Wire: Pi Pin 2 (5V) → DIR+ (or Pin 4)
  Wire: Pi Pin 6 (GND) → DIR-
  Loose wire: Pi Pin 6 (GND) — hold bare end in hand
  TAP loose wire against PUL- terminal rapidly
  50 taps at 400 pulses/rev ≈ 45° rotation
""")

input("Press Enter after completing manual tests (or Ctrl+C to abort)...")

print("\nNow running automated pulse test via GPIO.")
print("This sends 400 pulses = 1 full revolution at 400 steps/rev.")
print("Wiring for this test:")
print("  Pi Pin 2 (5V)  → PUL+")
print("  Pi Pin 2 (5V)  → DIR+  (or Pin 4)")
print("  Pi Pin 6 (GND) → DIR-")
print("  GPIO 17 (Pin 11) → 1K resistor → PUL-")
print()

try:
    import lgpio
except ImportError:
    print("ERROR: lgpio not available. Run this on the Pi.")
    sys.exit(1)

h = lgpio.gpiochip_open(0)
PUL = 17
lgpio.gpio_claim_output(h, PUL, 0, lgpio.SET_PULL_NONE)

print("Sending 400 pulses (1 full rev) at 3ms period...")
print("Watch the motor shaft!")
time.sleep(1)

for i in range(400):
    lgpio.gpio_write(h, PUL, 1)
    time.sleep(0.0015)
    lgpio.gpio_write(h, PUL, 0)
    time.sleep(0.0015)

print("Done. Did the motor complete one full revolution?")
print()
print("If YES → driver and motor confirmed working. Proceed to Phase 2.")
print("If NO  → check encoder cable, motor phase wiring (A+/A-, B+/B-),")
print("         and try swapping A+ and A- wires on the driver.")

lgpio.gpiochip_close(h)
