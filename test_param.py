from pymavlink import mavutil
import time

def set_param(master, param_name, param_value, param_type=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
    master.mav.param_set_send(
        master.target_system, master.target_component,
        param_name.encode('utf-8', 'ignore'),
        param_value,
        param_type
    )
    # Wait for acknowledgment
    start_time = time.time()
    while time.time() - start_time < 2:
        msg = master.recv_match(type='PARAM_VALUE', blocking=False)
        if msg:
            if msg.param_id.decode('utf-8').rstrip('\x00') == param_name:
                print(f"Successfully set {param_name} to {msg.param_value}")
                return True
    print(f"Failed to confirm {param_name} update")
    return False
