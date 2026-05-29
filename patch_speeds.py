with open("drone_control.py", "r") as f:
    text = f.read()

# Add setup_vertical_speeds
setup_func = """
def setup_vertical_speeds(speed_cm_s=25):
    '''Configures ArduPilot vertical params for slow, controlled takeoff and landing.'''
    speed = float(speed_cm_s)
    try:
        # Slow climb
        master.mav.param_set_send(master.target_system, master.target_component, b'WPNAV_SPEED_UP', speed, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        master.mav.param_set_send(master.target_system, master.target_component, b'WPNAV_ACCEL_Z', speed, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        master.mav.param_set_send(master.target_system, master.target_component, b'PILOT_SPEED_UP', speed, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        master.mav.param_set_send(master.target_system, master.target_component, b'PILOT_ACCEL_Z', speed, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        # Slow descent
        master.mav.param_set_send(master.target_system, master.target_component, b'WPNAV_SPEED_DN', speed, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        master.mav.param_set_send(master.target_system, master.target_component, b'LAND_SPEED', speed, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        master.mav.param_set_send(master.target_system, master.target_component, b'LAND_SPEED_HIGH', speed, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(0.1)
    except Exception as e:
        print(f"Warning: Could not set speeds: {e}")

def stabilize_position():
    '''Commands 0 velocity to eliminate drift before landing modes.'''
    print("Stabilizing (zeroing horizontal velocity) to prevent drift...")
    for _ in range(15): # Send command for 1.5 seconds
        send_body_ned_velocity(0, 0, 0)
        time.sleep(0.1)
"""

if "def setup_vertical_speeds" not in text:
    text = text.replace("def takeoff(altitude=1):", setup_func + "\ndef takeoff(altitude=1):")


# Patch takeoff
old_takeoff = """def takeoff(altitude=1):
    \"\"\"Commands the drone to take off to a specified altitude.\"\"\"
    print(f"Taking Off to {altitude} meters...")
    master.mav.command_long_send(master.target_system, master.target_component,
                                 mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                                 0, 0, 0, 0, 0, 0, altitude)
    # Wait for a sufficient time for the drone to reach takeoff altitude
    # This duration might need adjustment based on your simulation/drone's performance
    print(f"Waiting for takeoff to complete (approx. {altitude * 3} seconds)...") # Rough estimate
    time.sleep(altitude * 3) # Simple heuristic for takeoff time"""

new_takeoff = """def takeoff(altitude=1):
    \"\"\"Commands the drone to take off to a specified altitude.\"\"\"
    print("Pre-configuring safe takeoff speeds (25 cm/s)...")
    setup_vertical_speeds(25)
    
    print(f"Taking Off to {altitude} meters...")
    master.mav.command_long_send(master.target_system, master.target_component,
                                 mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                                 0, 0, 0, 0, 0, 0, altitude)
    
    wait_time = max(5, (altitude * 100) / 25.0 + 3)
    print(f"Waiting for takeoff to complete (approx. {wait_time:.1f} seconds)...")
    time.sleep(wait_time)"""
if old_takeoff in text:
    text = text.replace(old_takeoff, new_takeoff)

# Patch land_vehicle
old_land = """def land_vehicle():
    \"\"\"Commands the drone to land and waits until it has disarmed.\"\"\"
    print("Switching to LAND mode...")
    set_mode("LAND")
    print("Waiting for drone to land and disarm...")
    master.motors_disarmed_wait()  # This blocks until landing is complete
    print("Landed and disarmed.")"""

new_land = """def land_vehicle():
    \"\"\"Commands the drone to land and waits until it has disarmed.\"\"\"
    setup_vertical_speeds(25)
    stabilize_position()
    
    print("Switching to LAND mode...")
    set_mode("LAND")
    print("Waiting for drone to land and disarm...")
    master.motors_disarmed_wait()  # This blocks until landing is complete
    print("Landed and disarmed.")"""
if old_land in text:
    text = text.replace(old_land, new_land)

# Patch special_landing
old_special = """def special_landing():
    flush_data("RANGEFINDER")
    print("Switching to LAND mode...")
    set_mode("LAND")"""

new_special = """def special_landing():
    setup_vertical_speeds(25)
    stabilize_position()
    flush_data("RANGEFINDER")
    print("Switching to LAND mode...")
    set_mode("LAND")"""
if old_special in text:
    text = text.replace(old_special, new_special)


with open("drone_control.py", "w") as f:
    f.write(text)
print("Done patching drone_control.py!")
