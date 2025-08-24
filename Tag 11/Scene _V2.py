"""
Conveyor-Steuerung für Factory I/O (Modbus TCP)
Szenario: Ein Förderband, ein End-Sensor.
Logik:
- Wenn am Anfang eine Kiste erkannt wird (Startsensor ODER manuelles Startsignal), Band EIN.
- Wenn der End-Sensor die Kiste sieht, Band AUS.
- Mit einfacher Entprellung + sicherem Shutdown.

Passe unten DI_* und COIL_* an, falls deine Adressen abweichen.
"""

import time
from collections import deque
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

HOST = "127.0.0.1"
PORT = 502
POLL_MS = 100           # Abtastrate
DEBOUNCE = 3            # wie viele gleiche Reads in Folge für stabilen Zustand

# ==== Adressen aus Drivers-Fenster (0-basiert!) ====
# Häufige Defaults in Factory I/O:
DI_START_BOX   = 0      # Discrete Input: Sensor am Bandanfang (optional/falls vorhanden)
DI_END_BOX     = 1      # Discrete Input: Sensor am Bandende (orange Sensor im Bild)
COIL_CONVEYOR  = 0      # Coil: Motor Förderband EIN/AUS

def ensure_connected(client: ModbusTcpClient):
    if not client.connected:
        client.connect()

def read_di(client, address, count=1) -> bool:
    rr = client.read_discrete_inputs(address=address, count=count)
    if rr.isError():
        raise ModbusException(rr)
    return bool(rr.bits[0])

def read_coil(client, address, count=1) -> bool:
    rr = client.read_coils(address=address, count=count)
    if rr.isError():
        raise ModbusException(rr)
    return bool(rr.bits[0])

def write_coil(client, address, value: bool):
    rq = client.write_coil(address=address, value=value)
    if rq.isError():
        raise ModbusException(rq)

class DebouncedSignal:
    """Einfaches Entprellen über ein FIFO von bools."""
    def __init__(self, size=DEBOUNCE):
        self.buf = deque([False]*size, maxlen=size)

    def push(self, val: bool) -> bool:
        self.buf.append(val)
        # stabil, wenn alle gleich
        return all(self.buf) if any(self.buf) else False

def main():
    client = ModbusTcpClient(HOST, port=PORT, timeout=2)
    deb_start = DebouncedSignal()
    deb_end   = DebouncedSignal()

    conveyor_on = False
    last_start  = False  # für Flankenerkennung

    try:
        ensure_connected(client)
        print("Verbunden mit Factory I/O Modbus-Server.")

        while True:
            try:
                ensure_connected(client)

                # --- Sensoren lesen & entprellen ---
                start_raw = read_di(client, DI_START_BOX) if DI_START_BOX is not None else False
                end_raw   = read_di(client, DI_END_BOX)

                start_ok = deb_start.push(start_raw)
                end_ok   = deb_end.push(end_raw)

                # --- Flankenerkennung am Start (optional) ---
                rising_start = (start_ok and not last_start)
                last_start   = start_ok

                # --- Steuerlogik ---
                # Band einschalten, wenn neue Kiste am Anfang erkannt
                if rising_start and not conveyor_on:
                    conveyor_on = True

                # Band ausschalten, wenn Endsensor belegt
                if end_ok and conveyor_on:
                    conveyor_on = False

                # --- Aktor setzen (nur bei Änderung) ---
                if read_coil(client, COIL_CONVEYOR) != conveyor_on:
                    write_coil(client, COIL_CONVEYOR, conveyor_on)

                # Debug
                print(f"start:{int(start_ok)} end:{int(end_ok)} conveyor:{int(conveyor_on)}")

                time.sleep(POLL_MS/1000.0)

            except (ModbusException, OSError) as e:
                print(f"[WARN] Modbus-Problem: {e}. Reconnect…")
                time.sleep(0.5)
                ensure_connected(client)

    except KeyboardInterrupt:
        print("\nStop per KeyboardInterrupt.")
    finally:
        # Sicherheit: Band aus
        try:
            write_coil(client, COIL_CONVEYOR, False)
        except Exception:
            pass
        client.close()
        print("Verbindung geschlossen, Conveyor OFF.")

if __name__ == "__main__":
    main()
