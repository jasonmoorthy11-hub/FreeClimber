#!/usr/bin/env python3
import lgpio, time, subprocess, os, signal

# Kill any other python3 processes holding GPIO (except ourselves)
my_pid = os.getpid()
try:
    result = subprocess.run(['pgrep', '-f', 'python3'], capture_output=True, text=True)
    for pid_str in result.stdout.strip().split('\n'):
        if pid_str and int(pid_str) != my_pid:
            try:
                os.kill(int(pid_str), signal.SIGKILL)
            except:
                pass
except:
    pass

time.sleep(1)

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, 17, 0)  # Pin 11 = LOW
lgpio.gpio_claim_output(h, 27, 0)  # Pin 13 = LOW
print("GPIO 17 (pin 11) = LOW", flush=True)
print("GPIO 27 (pin 13) = LOW", flush=True)
print("Holding for 3 minutes. Probe now!", flush=True)
time.sleep(180)
lgpio.gpiochip_close(h)
print("Done.")
