# Transistor Level-Shifting Circuit — Breadboard Wiring Guide

## What You Need From the Smraza Kit
- 1× Breadboard
- 2× 2N2222 NPN transistors (small black plastic, flat side has text)
- 2× 1K resistors (color bands: Brown-Black-Red-Gold)
- 5× Male-to-Female DuPont jumper wires (for Pi GPIO header → breadboard)
- 2× Male-to-Male jumper wires (for breadboard → driver terminals)

---

## STEP 0: Identify the 2N2222 Transistor Pins

Hold the transistor with the **flat side facing you** and **legs pointing down**:

```
    FLAT SIDE (text visible)
    ___________________
   |                   |
   |     2N2222        |
   |___________________|
     |       |       |
     E       B       C

  LEFT    MIDDLE    RIGHT
 EMITTER   BASE   COLLECTOR
```

**This is CRITICAL — wrong orientation = circuit won't work or could damage components.**

---

## STEP 1: Orient the Breadboard

```
     A  B  C  D  E     F  G  H  I  J
     ┌──────────────┐  ┌──────────────┐
  1  │ · · · · ·    │  │    · · · · · │
  2  │ · · · · ·    │  │    · · · · · │
  3  │ · · · · ·    │  │    · · · · · │
  4  │ · · · · ·    │  │    · · · · · │
  5  │ · · · · ·    │  │    · · · · · │
  ...│              │  │              │
  30 │ · · · · ·    │  │    · · · · · │
     └──────────────┘  └──────────────┘
  +  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○   ← RED power rail (+)
  -  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○   ← BLUE ground rail (-)
```

**Key rules:**
- Each numbered ROW (1-30) in a half is connected horizontally (A-E or F-J)
- The + and - rails run the full length of the board
- The CENTER GAP separates left half (A-E) from right half (F-J)

---

## STEP 2: Place Transistors

Place both 2N2222 transistors straddling rows, **flat side facing you**:

```
  Transistor 1 (PULSE):     Transistor 2 (DIRECTION):
  Row 5, columns A-B-C      Row 10, columns A-B-C

     A    B    C                A    B    C
  5  [E]  [B]  [C]         10  [E]  [B]  [C]
     ↓    ↓    ↓               ↓    ↓    ↓
    EMT  BASE  COL            EMT  BASE  COL
```

Push the three legs into three adjacent holes in the SAME row.
Flat side faces you (toward the bottom edge of the breadboard).

---

## STEP 3: Connect Emitters to Ground

Both emitters (column A) need to connect to the GROUND rail (-):

```
  Wire 1: Row 5, column A  →  Ground rail (-)
  Wire 2: Row 10, column A →  Ground rail (-)
```

Use short jumper wires (male-to-male).

---

## STEP 4: Place the 1K Resistors

Each resistor connects the BASE to a nearby row where the Pi GPIO wire will go:

```
  Resistor 1: Row 5, column B (base of T1)  →  Row 5, column D (same row, other hole)
              Then from Row 8, column D         ← actually, use a DIFFERENT row:

  BETTER: Bridge rows with the resistor:
  Resistor 1: One leg in Row 5, col D  —  Other leg in Row 3, col D
  Resistor 2: One leg in Row 10, col D —  Other leg in Row 8, col D
```

Wait — let me give you a cleaner layout. Here's the exact placement:

---

## COMPLETE BREADBOARD LAYOUT

```
         A     B     C     D     E          F  G  H  I  J

  ROW 3  ·     ·     ·    [R1]   ·          ·  ·  ·  ·  ·    ← Resistor 1 end
                                                                 (Pi pin 11 wire here)
  ROW 4  ·     ·     ·     ·     ·          ·  ·  ·  ·  ·

  ROW 5  (E1)  (B1)  (C1) [R1]   ·          ·  ·  ·  ·  ·    ← TRANSISTOR 1
         │      │     │                                          (PUL circuit)
         │      └─ base gets signal via R1
         └─ emitter → GND rail                │
                      └─ collector → wire to PUL- on driver

  ROW 6  ·     ·     ·     ·     ·          ·  ·  ·  ·  ·

  ...

  ROW 8  ·     ·     ·    [R2]   ·          ·  ·  ·  ·  ·    ← Resistor 2 end
                                                                 (Pi pin 13 wire here)
  ROW 9  ·     ·     ·     ·     ·          ·  ·  ·  ·  ·

  ROW 10 (E2)  (B2)  (C2) [R2]   ·          ·  ·  ·  ·  ·    ← TRANSISTOR 2
         │      │     │                                          (DIR circuit)
         │      └─ base gets signal via R2
         └─ emitter → GND rail    │
                      └─ collector → wire to DIR- on driver

  GND(-) ════════════════════════════════════════════════════
  rail    ↑
          Connect to Pi pin 6 (GND)
```

---

## STEP-BY-STEP WIRING (Do in this exact order)

### Step A: Ground rail
```
  Pi Pin 6 (GND) ——[female-to-male DuPont wire]——→ Breadboard GND rail (-)
```

### Step B: Place Transistor 1 (PUL) in Row 5
```
  Flat side facing you. Legs into Row 5:
    Col A = Emitter
    Col B = Base
    Col C = Collector
```

### Step C: Transistor 1 emitter to ground
```
  Short jumper wire: Row 5, Col A ——→ GND rail (-)
```

### Step D: Resistor 1 (bridges Row 3 to Row 5)
```
  1K resistor (Brown-Black-Red-Gold):
    One leg → Row 5, Col D  (this connects to Base via the row)
    Other leg → Row 3, Col D
```
Note: Row 5 cols A-E are all connected, so col D in row 5 reaches the Base in col B.

### Step E: Pi GPIO 17 to Resistor 1
```
  Pi Pin 11 (GPIO 17) ——[female-to-male DuPont wire]——→ Row 3, Col E
```
Row 3 cols A-E are connected, so this reaches the resistor in col D.

### Step F: Collector 1 to driver PUL-
```
  Row 5, Col C ——[male-to-male jumper wire]——→ PUL- terminal on 2HSS86 driver
```
(Or use a longer wire if the driver is far from the breadboard)

### Step G: Place Transistor 2 (DIR) in Row 10
```
  Same as transistor 1. Flat side facing you. Legs into Row 10:
    Col A = Emitter
    Col B = Base
    Col C = Collector
```

### Step H: Transistor 2 emitter to ground
```
  Short jumper wire: Row 10, Col A ——→ GND rail (-)
```

### Step I: Resistor 2 (bridges Row 8 to Row 10)
```
  1K resistor:
    One leg → Row 10, Col D
    Other leg → Row 8, Col D
```

### Step J: Pi GPIO 27 to Resistor 2
```
  Pi Pin 13 (GPIO 27) ——[female-to-male DuPont wire]——→ Row 8, Col E
```

### Step K: Collector 2 to driver DIR-
```
  Row 10, Col C ——[male-to-male jumper wire]——→ DIR- terminal on 2HSS86 driver
```

### Step L: 5V power to driver signal inputs
```
  Pi Pin 2 (5V) ——[female-to-male DuPont wire]——→ PUL+ terminal on driver
  Pi Pin 4 (5V) ——[female-to-male DuPont wire]——→ DIR+ terminal on driver
```

---

## FINAL WIRING CHECKLIST

| From | To | Wire Type |
|---|---|---|
| Pi Pin 2 (5V) | Driver PUL+ | Female-to-male DuPont |
| Pi Pin 4 (5V) | Driver DIR+ | Female-to-male DuPont |
| Pi Pin 6 (GND) | Breadboard GND rail | Female-to-male DuPont |
| Pi Pin 11 (GPIO 17) | Breadboard Row 3, Col E | Female-to-male DuPont |
| Pi Pin 13 (GPIO 27) | Breadboard Row 8, Col E | Female-to-male DuPont |
| Row 5, Col A (T1 emitter) | GND rail | Short jumper |
| Row 10, Col A (T2 emitter) | GND rail | Short jumper |
| Row 5, Col C (T1 collector) | Driver PUL- | Jumper/wire |
| Row 10, Col C (T2 collector) | Driver DIR- | Jumper/wire |
| 1K Resistor 1 | Row 5 Col D ↔ Row 3 Col D | Resistor legs |
| 1K Resistor 2 | Row 10 Col D ↔ Row 8 Col D | Resistor legs |

**Total connections: 11 (5 from Pi, 2 to driver, 2 to GND rail, 2 resistors)**

---

## PI GPIO HEADER — FIND YOUR PINS

Looking at the Pi with the USB ports facing you (bottom), the GPIO header is at the top:

```
                    ┌─────────────────────────┐
                    │  Raspberry Pi 4          │
                    │                          │
   3.3V  (1) ●  ● (2)  5V  ◄── TO PUL+       │
  GPIO2  (3) ●  ● (4)  5V  ◄── TO DIR+       │
  GPIO3  (5) ●  ● (6)  GND ◄── TO BREADBOARD │
  GPIO4  (7) ●  ● (8)  GPIO14                 │
    GND  (9) ●  ● (10) GPIO15                 │
 GPIO17 (11) ●  ● (12) GPIO18                 │
    ▲                                          │
    └── TO RESISTOR 1 (PUL)                   │
 GPIO27 (13) ●  ● (14) GND                    │
    ▲                                          │
    └── TO RESISTOR 2 (DIR)                   │
 GPIO22 (15) ●  ● (16) GPIO23                 │
   3.3V (17) ●  ● (18) GPIO24                 │
            ...                                │
                    │          ┌──┐ ┌──┐       │
                    │          │USB│ │USB│      │
                    └──────────┴──┘─┴──┘───────┘
```

**You only use 5 pins: 2, 4, 6, 11, 13** (all on the left column except pin 4)

---

## HOW IT WORKS (Signal Flow)

```
  Pi GPIO 17 ──→ 1K Resistor ──→ Base of 2N2222
                                        │
  GPIO HIGH (3.3V)                      ▼
  = transistor turns ON          Collector ──→ PUL- on driver
  = collector pulled to GND             │
  = 5V across opto (PUL+ is 5V)        │
  = PULSE ON                     Emitter ──→ GND

  GPIO LOW (0V)
  = transistor turns OFF
  = collector floats (opto sees ~0V)
  = PULSE OFF
```

Each HIGH→LOW cycle on GPIO 17 = one motor step.

---

## TROUBLESHOOTING

If motor doesn't move after wiring:
1. Check transistor orientation (flat side = E, B, C left to right)
2. Check resistor color bands (Brown-Black-Red-Gold = 1K)
3. Verify with multimeter: measure between PUL+ and PUL- at driver
   - Should see ~5V when GPIO HIGH, ~0V when GPIO LOW
4. Swap transistors (in case one is bad)
5. Check driver green light (no alarm)
6. Run: `python3 ~/motor_control/motor_control.py cw 200 0.002`
