# Import mavutil
from pymavlink import mavutil
import time
from drone_control import *

# Create the connection
# Need to provide the serial port and baudrate
# master = mavutil.mavlink_connection("/dev/tty.usbserial-D30JKVZM", baud=57600)
connect_to_vehicle()
get_safe_spot_pos()

# while True:
# #     msg = master.recv_match(blocking=True, type='GLOBAL_POSITION_INT').to_dict()
# #     print(int(msg['relative_alt'])/1000)
# #     msg = master.recv_match(blocking=True, type='LOCAL_POSITION_NED').to_dict()
# #     print(f"x: {msg['x']}, y: {msg['y']}, z: {msg['z']}")
#     msg = master.recv_match(blocking=True, type='RANGEFINDER').to_dict()
#     print(f"Distance: {msg['distance']} m")
#     time.sleep(0.1)