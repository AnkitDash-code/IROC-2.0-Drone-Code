import json
from pymavlink import mavutil
import time

# Create the connection
# Need to provide the serial port and baudrate

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
print("Trying to find heartbeat...")
master.wait_heartbeat()
print("found it")


# Arm
# master.arducopter_arm() or:
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1, 0, 0, 0, 0, 0, 0)

# wait until arming confirmed (can manually check with master.motors_armed())
print("Waiting for the vehicle to arm")
master.motors_armed_wait()
print('Armed!')

time.sleep(20)

# Disarm
# master.arducopter_disarm() or:
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    0, 0, 0, 0, 0, 0, 0)

master.motors_disarmed_wait()

n=1
while n != 5:
    try:
        msg = master.recv_match().to_dict()
        with open('file.json', 'w') as file:
            json.dump(msg, file, indent=4)
        print(msg)
    except:
        print("...")
        pass
    time.sleep(0.1)