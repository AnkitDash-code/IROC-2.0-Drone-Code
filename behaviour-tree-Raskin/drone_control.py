"""Lightweight mock of drone_control for local testing.
This module provides minimal functions used by the BT nodes so tests
can run without real MAVLink/SITL.
"""

master = None

_state = {
    "mode": None,
    "armed": False,
}

def set_mode(mode: str):
    _state["mode"] = mode
    print(f"[mock dc] set_mode({mode})")

def arm_vehicle():
    _state["armed"] = True
    print("[mock dc] arm_vehicle()")

def rotate_towards(x, y):
    print(f"[mock dc] rotate_towards -> ({x},{y})")
    # return a mock distance
    return 1.0

def move_body_ned(dist, speed_mps=0.5):
    print(f"[mock dc] move_body_ned(dist={dist}, speed={speed_mps})")

def send_body_ned_velocity(vx, vy, vz):
    print(f"[mock dc] send_body_ned_velocity(vx={vx}, vy={vy}, vz={vz})")

def takeoff(alt_m: float):
    print(f"[mock dc] takeoff({alt_m}m)")

def get_state():
    return dict(_state)

def connect_to_vehicle(timeout=30):
    """Open a MAVLink connection to the vehicle using `CONNECTION_STRING` and `BAUD_RATE`.
    This mirrors the behavior expected by `ConnectVehicle` so the BT can use `dc.master`.
    """
    global master
    try:
        from pymavlink import mavutil
    except Exception as e:
        raise RuntimeError(f"pymavlink not available: {e}")

    conn = globals().get('CONNECTION_STRING', None)
    baud = globals().get('BAUD_RATE', None)
    if not conn:
        raise RuntimeError("CONNECTION_STRING not set on drone_control module")
    if not baud:
        baud = 57600

    try:
        master = mavutil.mavlink_connection(conn, baud=baud)
        master.wait_heartbeat(timeout=timeout)
    except Exception as e:
        master = None
        raise RuntimeError(f"Failed to connect to vehicle at {conn}: {e}")

    return master
