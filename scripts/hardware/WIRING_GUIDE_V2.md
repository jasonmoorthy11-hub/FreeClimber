# Wiring Guide V2 — 2HSS86 + NEMA 34 + Raspberry Pi

## DIP Switch Settings (SET THESE FIRST)

On the side of the 2HSS86 driver, 8 DIP switches:

| Switch | Position | Setting |
|--------|----------|---------|
| SW1 | ON | 6.0A peak current |
| SW2 | OFF | (current setting) |
| SW3 | OFF | (current setting) |
| SW4 | OFF | Full holding torque |
| SW5 | ON | 400 pulses/rev |
| SW6 | ON | (microstep setting) |
| SW7 | ON | (microstep setting) |
| SW8 | ON | (microstep setting) |

**Power cycle the 70V PSU after changing DIP switches.**

At 400 pulses/rev, each pulse = 0.9 degrees (visible movement).

---

## Option A: Direct Sink Wiring (RECOMMENDED)

**Simplest possible wiring. No transistor, no breadboard, no resistor.**

The Pi GPIO sinks ~10mA from 5V through the driver's internal opto-LED.
Logic is INVERTED: GPIO LOW = pulse, GPIO HIGH = idle.

### Connections (4 wires total)

| From | To | Wire |
|------|----|------|
| Pi Pin 2 (5V) | Driver PUL+ | M-to-F jumper |
| Pi Pin 4 (5V) | Driver DIR+ | M-to-F jumper |
| Pi GPIO 17 (Pin 11) | Driver PUL- | M-to-F jumper (DIRECT) |
| Pi GPIO 27 (Pin 13) | Driver DIR- | M-to-F jumper (DIRECT) |

```
Pi GPIO Header (USB ports facing away):

  (1)  3.3V    ●  ●  [5V]   (2)  → PUL+
  (3)  GPIO2   ●  ●  [5V]   (4)  → DIR+
  (5)  GPIO3   ●  ●   GND   (6)
  (7)  GPIO4   ●  ●   TX    (8)
  (9)  GND     ●  ●   RX    (10)
  (11)[GPIO17] ●  ●  GPIO18 (12)  → PUL- (direct)
  (13)[GPIO27] ●  ●   GND   (14)  → DIR- (direct)
```

### How It Works

```
5V (Pin 2) ──→ PUL+ ──→ [opto-LED inside driver] ──→ PUL- ──→ GPIO 17
                         (internal 330Ω resistor)

When GPIO = LOW:  current flows 5V → opto → GPIO sink = PULSE ON
When GPIO = HIGH: no current flow = PULSE OFF

Current = (5V - 1.2V - 0.4V) / 330Ω ≈ 10.3mA ✓
```

### Software Mode

Use `wiring_mode="sink"` in motor.py (this is the default):

```python
motor = RINGMotor(wiring_mode="sink", steps_per_rev=400)
```

### Quick Test

```bash
ssh flyclimberpi@flyclimberpi.local "python3 ~/motor.py"
```

Or manually:

```bash
ssh flyclimberpi@flyclimberpi.local "python3 -c \"
import lgpio, time
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, 17, 1, lgpio.SET_PULL_NONE)  # idle HIGH
lgpio.gpio_claim_output(h, 27, 1, lgpio.SET_PULL_NONE)
lgpio.gpio_write(h, 27, 0)  # direction active
time.sleep(0.01)
for i in range(400):  # 1 full rev at 400 steps/rev
    lgpio.gpio_write(h, 17, 0)  # pulse ON (inverted!)
    time.sleep(0.0015)
    lgpio.gpio_write(h, 17, 1)  # pulse OFF
    time.sleep(0.0015)
lgpio.gpiochip_close(h)
print('Done - 1 full revolution')
\""
```

---

## Option B: Transistor Circuit Wiring (Alternative)

Use this if you need electrical isolation between the Pi and driver,
or if direct sink wiring doesn't provide enough current for your driver.

Requires: 2× 2N2222 transistors, 2× 1K resistors, breadboard.

### Transistor Pinout

```
     FLAT SIDE (text visible)
      ___________________
     |     2N2222        |
     |___________________|
       |       |       |
       E       B       C
     LEFT   MIDDLE   RIGHT
```

### Breadboard Layout

```
     a     b     c     d     e
  3  ·     ·     ·    [R1]  [←Pi Pin 11 (GPIO17)]
  4  ·     ·     ·     ·     ·
  5  [E1]  [B1]  [C1] [R1]  [→PUL- green wire]
     │
     └──→ GND rail → Pi Pin 6

  13 ·     ·     ·    [R2]  [←Pi Pin 13 (GPIO27)]
  14 ·     ·     ·     ·     ·
  15 [E2]  [B2]  [C2] [R2]  [→DIR- green wire]
     │
     └──→ GND rail (same rail)
```

### All Connections

| # | From | To | Wire |
|---|------|----|------|
| 1 | Transistor 1 | Row 5 (a=E, b=B, c=C) | component |
| 2 | Transistor 2 | Row 15 (a=E, b=B, c=C) | component |
| 3 | Row 5 col a (T1 emitter) | GND rail | short jumper |
| 4 | Row 15 col a (T2 emitter) | GND rail | short jumper |
| 5 | Resistor 1 | Row 5 col d ↔ Row 3 col d | resistor legs |
| 6 | Resistor 2 | Row 15 col d ↔ Row 13 col d | resistor legs |
| 7 | Pi Pin 6 (GND) | Breadboard GND rail | M-to-F (black) |
| 8 | Pi Pin 11 (GPIO17) | Row 3 col e | M-to-F |
| 9 | Pi Pin 13 (GPIO27) | Row 13 col e | M-to-F |
| 10 | Driver PUL- (green) | Row 5 col e | from driver |
| 11 | Driver DIR- (green) | Row 15 col e | from driver |
| 12 | Driver PUL+ (black) | Pi Pin 2 (5V) | from driver |
| 13 | Driver DIR+ (blue) | Pi Pin 4 (5V) | from driver |

### Software Mode

Use `wiring_mode="source"` (normal logic: HIGH = pulse):

```python
motor = RINGMotor(wiring_mode="source", steps_per_rev=400)
```

---

## Debugging Checklist

If motor doesn't move, work through these in order:

1. **DIP switches correct?** (SW5-8 all ON for 400 steps/rev)
2. **Power cycled after DIP change?**
3. **Shaft locked?** (try to turn by hand — should resist strongly)
4. **Opto-LED alive?** (diode mode: PUL+ to PUL- = 1.0-1.2V)
5. **Manual tap test?** (5V→PUL+, tap GND wire to PUL-, motor should step)
6. **Run phase diagnostics:** `python3 ~/diagnostics/phase1_driver_test.py`

### Diagnostic Scripts

Located in `scripts/hardware/diagnostics/`:

| Script | Tests |
|--------|-------|
| `phase1_driver_test.py` | Driver/motor in isolation (manual + automated) |
| `phase2a_direct_source.py` | GPIO source drive (likely too weak) |
| `phase2b_direct_sink.py` | GPIO sink drive (recommended, ~10mA) |
| `phase3_transistor_debug.py` | Transistor circuit voltage checks |
