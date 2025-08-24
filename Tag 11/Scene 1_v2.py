# -*- coding: utf-8 -*-
import time
import csv
from collections import deque
from pathlib import Path
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

# ========= Verbindung =========
HOST = "192.168.56.1"   # -> nimm "127.0.0.1", wenn du im Driver localhost gewählt hast
PORT = 502              # -> oft einfacher: 1502 im Driver + hier 1502

POLL_MS = 100
RETRY_SLEEP = 1.0

# ========= CSV =========
ENABLE_CSV = True
CSV_PATH = Path("box_log.csv")

# ========= Mapping laut Szene =========
# Discrete Inputs (read-only)
DI_ITEM_READY = 0
DI_AT_ENTRY   = 1
DI_AT_EXIT    = 2
DI_RUNNING    = 3

# Coils (write/read)
COIL_ENTRY_CONVEYOR  = 1      # Förderband 1
COIL_BUFFER_CONVEYOR = 0     # Förderband 2
COIL_EMITTER = 3
# (Optional) – NUR setzen, wenn im Driver verdrahtet:
COIL_RUN   =0            # z. B. 2  -> "FACTORY I/O (Run)"
COIL_PAUSE = None             # z. B. 3  -> "FACTORY I/O (Pause)"
COIL_RESET = None             # z. B. 4  -> "FACTORY I/O (Reset)"

# (Eine der beiden Varianten benutzen, je nachdem was du verdrahtet hast)
COIL_SPAWN        = 1      # z. B. 5  -> "Source/Spawn" (pro Puls 1 Kiste erzeugen)
COIL_STOPPER_OPEN = None      # z. B. 6  -> "Stopper Open" (kurz öffnen = 1 Kiste durchlassen)

# ========= Rate-Regelung =========
# Wähle "spawn", "stopper" oder None (aus)
RATE_MODE   = "spawn"           # "spawn" ODER "stopper" ODER None
TARGET_PPM  = 10              # Teile pro Minute (nur wenn RATE_MODE gesetzt)
OPEN_MS     = 220             # Öffnungsdauer für Stopper-Puls (tunen)
SPAWN_MS    = 120             # Pulsdauer für Spawn (tunen)

# ========= Entprellung =========
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
    def read_time(self):
        # read 1 input register starting at address 0
        rr = self.client.read_input_registers(address=0, count=1)
        
        if not rr.isError():
            raw_value = rr.registers[0]
            print("Register 0:", raw_value)
            return raw_value
        else:
            print("Error:", rr)
            return -1
def ensure_csv(path: Path):
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp_iso", "box_count"])
def log_box(path: Path, count: int):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), count])


# ------ Helpers ------
def pulse(bus: Bus, coil_addr: int, ms: int):
    """TRUE kurz setzen → FALSE."""
    bus.write_coil(coil_addr, True)
    time.sleep(ms / 1000.0)
    bus.write_coil(coil_addr, False)

def ensure_running(bus: Bus):
    """Wenn COIL_RUN verdrahtet ist: RUN flanken, bis DI_RUNNING==1."""
    if COIL_RUN is None:
        return
    try:
        rr = bus.read_di(DI_RUNNING, 1)[0]
    except Exception:
        rr = False
    if not rr:
        pulse(bus, COIL_RUN, 150)
        # warten bis Running hochkommt (max 2 s)
        for _ in range(20):
            try:
                if bus.read_di(DI_RUNNING, 1)[0]:
                    break
            except Exception:
                pass
            time.sleep(0.1)

def main():
    # CSV
    if ENABLE_CSV: ensure_csv(CSV_PATH)

    # Verbindung
    bus = Bus(HOST, PORT)
    while not bus.connect():
        print(f"❌ Connect {HOST}:{PORT} fehlgeschlagen – retry in {RETRY_SLEEP}s …")
        time.sleep(RETRY_SLEEP)
    print("✅ Verbunden. Szene RUN & FORCED ausschalten, wo du per Modbus steuern willst.")

    # Auto-RUN (falls verdrahtet)
    ensure_running(bus)
    
    # Bänder an (Dauerbetrieb)
    try:
        bus.write_coil(COIL_ENTRY_CONVEYOR, True)
    except Exception:
        pass
    try:
        bus.write_coil(COIL_BUFFER_CONVEYOR, True)
    except Exception:
        pass

    # Rate-Modus vorbereiten
    if RATE_MODE in ("spawn", "stopper"):
        interval_s = 60.0 / max(1, TARGET_PPM)
        next_release = time.monotonic() + 0.2  # kleiner Offset

    # Zähler/Entprellung
    deb_entry, deb_exit = DebounceHL(), DebounceHL()
    last_entry = last_exit = False
    conveyor_on = True           # Band läuft dauerhaft (wir „gaten“ davor)
    last_written = True          # weil oben bereits eingeschaltet

    box_count = 0

    try:
        while True:
           # bus.read_time()
           # bus.write_coil(1, True)
            t0 = time.perf_counter()
            try:
                # Optional: globale RUN-Überwachung
                running = bus.read_di(DI_RUNNING, 1)[0] if DI_RUNNING is not None else True

                # Rate-Impuls
                if RATE_MODE == "spawn" and (COIL_SPAWN is not None):
                    now = time.monotonic()
                    if now >= next_release:
                        pulse(bus, COIL_SPAWN, SPAWN_MS)
                        next_release += interval_s

                elif RATE_MODE == "stopper" and (COIL_STOPPER_OPEN is not None):
                    now = time.monotonic()
                    if now >= next_release:
                        pulse(bus, COIL_STOPPER_OPEN, OPEN_MS)
                        next_release += interval_s

                # Sensoren (entprellt)
                entry_ok = deb_entry.push(bus.read_di(DI_AT_ENTRY, 1)[0])
                exit_ok  = deb_exit.push(bus.read_di(DI_AT_EXIT, 1)[0])

                # Flanken
                rising_entry = entry_ok and not last_entry
                rising_exit  = exit_ok  and not last_exit
                last_entry, last_exit = entry_ok, exit_ok

                # Wenn du zusätzlich „Band auf Nachfrage“ willst, entkommentieren:
                # if rising_entry and not conveyor_on:
                #     conveyor_on = True
                # if rising_exit and conveyor_on:
                #     conveyor_on = False
                #     box_count += 1
                #     if ENABLE_CSV: log_box(CSV_PATH, box_count)

                # Hier zählen wir unabhängig vom Band: jede Box, die den Exit triggert
                if rising_exit:
                    box_count += 1
                    if ENABLE_CSV: log_box(CSV_PATH, box_count)

                # Bandzustand nur schreiben, wenn du „auf Nachfrage“ steuerst
                # (in diesem Template läuft das Band dauerhaft)
                if conveyor_on != last_written:
                    bus.write_coil(COIL_ENTRY_CONVEYOR, conveyor_on)
                    last_written = conveyor_on

                coil_entry = bus.read_coil(COIL_ENTRY_CONVEYOR, 1)[0]
                print(f"entry:{int(entry_ok)} exit:{int(exit_ok)} running:{int(running)} | "
                      f"coil:{int(coil_entry)} | count:{box_count}")

            except (ModbusException, OSError) as e:
                print(f"[WARN] Modbus: {e} → reconnect …")
                time.sleep(RETRY_SLEEP)
                bus.close()
                while not bus.connect():
                    print(f"[RETRY] connect {HOST}:{PORT} …")
                    time.sleep(RETRY_SLEEP)

            # Takt einhalten
            dt = (time.perf_counter() - t0)
            sleep_left = (POLL_MS/1000.0) - dt
            if sleep_left > 0:
                time.sleep(sleep_left)

    except KeyboardInterrupt:
        print("\n⏹ Stop.")
    finally:
        try: bus.write_coil(COIL_ENTRY_CONVEYOR, False)
        except Exception: pass
        try: bus.write_coil(COIL_BUFFER_CONVEYOR, False)
        except Exception: pass
        bus.close()
        print(f"Verbindung geschlossen. Conveyor OFF. Endcount={box_count}")
        if ENABLE_CSV: print(f"CSV: {CSV_PATH.resolve()}")

if __name__ == "__main__":
    main()
