#!/usr/bin/env python3
"""
Diagnostic: Try different pulse methods to find what triggers the 2HSS86.
"""

import lgpio
import time

PUL_PLUS = 11
PUL_MINUS = 9
DIR_PLUS = 13
DIR_MINUS = 6

h = lgpio.gpiochip_open(0)
for pin in [PUL_PLUS, PUL_MINUS, DIR_PLUS, DIR_MINUS]:
    lgpio.gpio_claim_output(h, pin, 0)

def pulse_n(steps, method, delay=0.002):
    for _ in range(steps):
        method(True)
        time.sleep(delay)
        method(False)
        time.sleep(delay)

print("=== 2HSS86 Pulse Diagnostic ===")
print("Watch/listen for motor movement after each test.\n")

# Set direction for all tests
lgpio.gpio_write(h, DIR_PLUS, 1)
lgpio.gpio_write(h, DIR_MINUS, 0)
time.sleep(0.01)

# Test 1: Original method (PUL+ high/low, PUL- stays low)
print("[Test 1] PUL+=toggle, PUL-=LOW (original)...")
time.sleep(1)
def m1(on):
    lgpio.gpio_write(h, PUL_PLUS, 1 if on else 0)
    lgpio.gpio_write(h, PUL_MINUS, 0)
pulse_n(200, m1)
print("  Did it move? Continuing in 2s...\n")
time.sleep(2)

# Test 2: Inverted (PUL+ stays low, PUL- toggles)
print("[Test 2] PUL+=LOW, PUL-=toggle (inverted)...")
time.sleep(1)
def m2(on):
    lgpio.gpio_write(h, PUL_PLUS, 0)
    lgpio.gpio_write(h, PUL_MINUS, 1 if on else 0)
pulse_n(200, m2)
print("  Did it move? Continuing in 2s...\n")
time.sleep(2)

# Test 3: Full differential (PUL+ and PUL- opposite)
print("[Test 3] PUL+ and PUL- fully differential (opposite states)...")
time.sleep(1)
def m3(on):
    lgpio.gpio_write(h, PUL_PLUS, 1 if on else 0)
    lgpio.gpio_write(h, PUL_MINUS, 0 if on else 1)
pulse_n(200, m3)
print("  Did it move? Continuing in 2s...\n")
time.sleep(2)

# Test 4: Swap PUL and DIR pins entirely (in case they're swapped)
print("[Test 4] Using DIR pins as PUL (in case wiring is swapped)...")
time.sleep(1)
# Use DIR pins for pulsing, PUL pins for direction
lgpio.gpio_write(h, PUL_PLUS, 1)
lgpio.gpio_write(h, PUL_MINUS, 0)
time.sleep(0.01)
def m4(on):
    lgpio.gpio_write(h, DIR_PLUS, 1 if on else 0)
    lgpio.gpio_write(h, DIR_MINUS, 0)
pulse_n(200, m4)
print("  Did it move? Continuing in 2s...\n")
time.sleep(2)

# Test 5: Longer pulses (maybe timing is too fast)
print("[Test 5] Very slow pulses on PUL+ (50 steps, 10Hz)...")
time.sleep(1)
lgpio.gpio_write(h, DIR_PLUS, 1)
lgpio.gpio_write(h, DIR_MINUS, 0)
time.sleep(0.01)
def m5(on):
    lgpio.gpio_write(h, PUL_PLUS, 1 if on else 0)
    lgpio.gpio_write(h, PUL_MINUS, 0)
pulse_n(50, m5, delay=0.05)
print("  Did it move? Continuing in 2s...\n")
time.sleep(2)

# Test 6: All 4 pins toggling individually
print("[Test 6] Toggling each GPIO individually (50 pulses each)...")
for pin_name, pin in [("GPIO6 (DIR-)", DIR_MINUS), ("GPIO9 (PUL-)", PUL_MINUS),
                       ("GPIO11 (PUL+)", PUL_PLUS), ("GPIO13 (DIR+)", DIR_PLUS)]:
    # Reset all low
    for p in [PUL_PLUS, PUL_MINUS, DIR_PLUS, DIR_MINUS]:
        lgpio.gpio_write(h, p, 0)
    time.sleep(0.1)
    print(f"  Toggling {pin_name}...")
    for _ in range(50):
        lgpio.gpio_write(h, pin, 1)
        time.sleep(0.02)
        lgpio.gpio_write(h, pin, 0)
        time.sleep(0.02)
    time.sleep(1)

print("\n=== Diagnostic Complete ===")
print("Which test (1-6) caused movement?")
print("If none worked, the 3.3V GPIO likely can't drive the optocouplers.")
print("Solution: wire PUL- and DIR- to GND, and connect PUL+ and DIR+")
print("through a level shifter (3.3V -> 5V) or add external resistors.")

# Cleanup
for pin in [PUL_PLUS, PUL_MINUS, DIR_PLUS, DIR_MINUS]:
    lgpio.gpio_write(h, pin, 0)
lgpio.gpiochip_close(h)
