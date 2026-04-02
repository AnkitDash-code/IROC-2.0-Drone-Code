with open("guarded_mission.py", "r") as f:
    text = f.read()

# Replace takeoff constants
old_constants = """SLOW_TAKEOFF_SPEED_UP = 10.0
SLOW_TAKEOFF_ACCEL_Z = 4.0"""

new_constants = """SLOW_TAKEOFF_SPEED_UP = 25.0
SLOW_TAKEOFF_ACCEL_Z = 20.0
SLOW_LANDING_SPEED = 25.0"""
text = text.replace(old_constants, new_constants)

# Replace set_slow_takeoff_profile logic completely
old_set_slow = """    def set_slow_takeoff_profile(self) -> None:
        \"\"\"
        Temporarily reduce vertical speed/acceleration for a smooth slow climb.
        \"\"\"
        for name, slow_value in (("WPNAV_SPEED_UP", SLOW_TAKEOFF_SPEED_UP), ("WPNAV_ACCEL_Z", SLOW_TAKEOFF_ACCEL_Z)):
            current = self.get_param(name)
            if current is not None:
                self._saved_params[name] = current
                print(f"[INFO] Saved {name}={current:.2f}")
            self.set_param(name, slow_value)"""

new_set_slow = """    def set_slow_takeoff_profile(self) -> None:
        \"\"\"
        Temporarily reduce vertical speed/acceleration for a smooth slow climb.
        \"\"\"
        speeds = [
            ("WPNAV_SPEED_UP", SLOW_TAKEOFF_SPEED_UP),
            ("WPNAV_ACCEL_Z", SLOW_TAKEOFF_ACCEL_Z),
            ("PILOT_SPEED_UP", SLOW_TAKEOFF_SPEED_UP),
            ("PILOT_ACCEL_Z", SLOW_TAKEOFF_ACCEL_Z)
        ]
        for name, slow_value in speeds:
            current = self.get_param(name)
            if current is not None and name not in self._saved_params:
                self._saved_params[name] = current
                print(f"[INFO] Saved {name}={current:.2f}")
            self.set_param(name, slow_value)

    def set_slow_landing_profile(self) -> None:
        \"\"\"
        Temporarily reduce vertical speed for a smooth descent.
        \"\"\"
        speeds = [
            ("WPNAV_SPEED_DN", SLOW_LANDING_SPEED),
            ("LAND_SPEED", SLOW_LANDING_SPEED),
            ("LAND_SPEED_HIGH", SLOW_LANDING_SPEED)
        ]
        for name, slow_value in speeds:
            current = self.get_param(name)
            if current is not None and name not in self._saved_params:
                self._saved_params[name] = current
                print(f"[INFO] Saved {name}={current:.2f}")
            self.set_param(name, slow_value)

    def stabilize_position(self) -> None:
        \"\"\"Commands 0 velocity to eliminate drift before landing modes.\"\"\"
        print("[INFO] Stabilizing (zeroing horizontal velocity) to prevent drift...")
        for _ in range(15):
            self.send_body_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)"""
text = text.replace(old_set_slow, new_set_slow)

# Replace land_and_disarm
old_land = """    def land_and_disarm(self) -> None:
        \"\"\"
        Use drone_control.land_vehicle() which waits for proper auto-disarm.
        \"\"\"
        self._sync_master()
        if self.master is None:
            raise RuntimeError("Vehicle not connected.")
        dc.land_vehicle()"""

new_land = """    def land_and_disarm(self) -> None:
        \"\"\"
        Use drone_control.land_vehicle() which waits for proper auto-disarm.
        \"\"\"
        self._sync_master()
        if self.master is None:
            raise RuntimeError("Vehicle not connected.")
            
        print("[INFO] Preparing safe landing profile (25 cm/s)..")
        self.set_slow_landing_profile()
        self.stabilize_position()
        
        dc.land_vehicle()"""
text = text.replace(old_land, new_land)

with open("guarded_mission.py", "w") as f:
    f.write(text)
print("Patch successful!")
