# -*- coding: utf-8 -*-
import time
import csv
from collections import deque
from pathlib import Path
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

HOST = "192.168.56.1"
PORT = 502
POLL_MS = 100
RETRY_SLEEP = 1.0
ENABLE_CSV = False
CSV_PATH = Path("box_log.csv")

# Mapping aus deiner Szene
DI_ITEM_READY = 0
DI_AT_ENTRY   = 1
DI_AT_EXIT    = 2
DI_RUNNING    = 3

COIL_ENTRY_CONVEYOR  = 0
COIL_BUFFER_CONVEYOR = 1

DEBOUNCE_HIGH = 3
DEBOUNCE_LOW  = 3

class DebounceHL:
    def __init__(self, high=DEBOUNCE_HIGH, low=DEBOUNCE_LOW):
        self.buf = deque([], maxlen=max(high, low))
        self.high, self.low = high, low
        self.stable = False
    def push(self, v: bool) -> bool:
        self.buf.append(bool(v))
        if len(self.buf) >= self.high and all(list(self.buf)[-self.high:]):
            self.stable = True
        elif len(self.buf) >= self.low and not any(list(self.buf)[-self.low:]):
            self.stable = False
        return self.stable

class Bus:
    def __init__(self, host, port, timeout=2.0):
        self.client = ModbusTcpClient(host, port=port, timeout=timeout)
    def connect(self): return self.client.connect()
    def close(self): 
        try: self.client.close()
        except Exception: pass
    def read_di(self, address, count=1):
        rr = self.client.read_discrete_inputs(address=address, count=count)
        if rr.isError(): raise ModbusException(rr)
        return rr.bits
    def read_coil(self, address, count=1):
        rr = self.client.read_coils(address=address, count=count)
        if rr.isError(): raise ModbusException(rr)
        return rr.bits
    def write_coil(self, address, value: bool):
        rq = self.client.write_coil(address=address, value=bool(value))
        if rq.isError(): raise ModbusException(rq)
        return True

def ensure_csv(path: Path):
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp_iso", "box_count"])
def log_box(path: Path, count: int):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), count])

def main():
    if ENABLE_CSV: ensure_csv(CSV_PATH)
    bus = Bus(HOST, PORT)
    while not bus.connect():
        print(f"❌ Connect {HOST}:{PORT} fehlgeschlagen – retry in {RETRY_SLEEP}s …")
        time.sleep(RETRY_SLEEP)
    print("✅ Verbunden. Stelle sicher: Factory I/O RUN, Slave ID = 0.")

    # Selbsttest: Coil toggeln
    try:
        print("🔎 Selbsttest: Conveyor ON/OFF …")
        bus.write_coil(COIL_ENTRY_CONVEYOR, True); time.sleep(0.2)
        on_state = bus.read_coil(COIL_ENTRY_CONVEYOR, 1)[0]
        bus.write_coil(COIL_ENTRY_CONVEYOR, False); time.sleep(0.2)
        off_state = bus.read_coil(COIL_ENTRY_CONVEYOR, 1)[0]
        print(f"   coil after ON={int(on_state)}, after OFF={int(off_state)}")
    except Exception as e:
        print(f"⚠️ Selbsttest fehlgeschlagen: {e}")

    deb_entry, deb_exit = DebounceHL(), DebounceHL()
    last_entry = last_exit = False
    conveyor_on = False
    last_written = False
    box_count = 0
    # Start the program
    bus.write_coil(0,True)
    bus.write_coil(2,True)
    bus.write_coil(1,True)
    bus.write_coil(3,True)
    try:
        while True:
            t0 = time.perf_counter()
            try:
                running  = bus.read_di(DI_RUNNING, 1)[0]
                entry_ok = deb_entry.push(bus.read_di(DI_AT_ENTRY, 1)[0])
                exit_ok  = deb_exit.push(bus.read_di(DI_AT_EXIT, 1)[0])

                if not running and conveyor_on:
                    conveyor_on = False

                rising_entry = entry_ok and not last_entry
                rising_exit  = exit_ok  and not last_exit
                last_entry, last_exit = entry_ok, exit_ok

                if rising_entry and not conveyor_on:
                    conveyor_on = True
                if rising_exit and conveyor_on:
                    conveyor_on = False
                    box_count += 1
                    if ENABLE_CSV: log_box(CSV_PATH, box_count)

                if conveyor_on != last_written:
                    bus.write_coil(COIL_ENTRY_CONVEYOR, conveyor_on)
                    last_written = conveyor_on

                coil_entry = bus.read_coil(COIL_ENTRY_CONVEYOR, 1)[0]
                print(f"entry:{int(entry_ok)} exit:{int(exit_ok)} running:{int(running)} | "
                      f"coil:{int(coil_entry)} | on:{int(conveyor_on)} count:{box_count}")

            except (ModbusException, OSError) as e:
                print(f"[WARN] Modbus: {e} → reconnect …")
                time.sleep(RETRY_SLEEP)
                bus.close()
                while not bus.connect():
                    print(f"[RETRY] connect {HOST}:{PORT} …")
                    time.sleep(RETRY_SLEEP)

            dt = (time.perf_counter() - t0)
            sleep_left = (POLL_MS/1000.0) - dt
            if sleep_left > 0: time.sleep(sleep_left)

    except KeyboardInterrupt:
        print("\n⏹ Stop.")
    finally:
        try: bus.write_coil(COIL_ENTRY_CONVEYOR, False)
        except Exception: pass
        bus.close()
        print(f"Verbindung geschlossen. Conveyor OFF. Endcount={box_count}")
        if ENABLE_CSV: print(f"CSV: {CSV_PATH.resolve()}")

if __name__ == "__main__":
    main()
