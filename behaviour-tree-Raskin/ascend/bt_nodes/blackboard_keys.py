# Single place that defines every key stored on the BT blackboard.
# This prevents typo-driven bugs where two nodes use different key names.

ALTITUDE_M = "/drone/altitude_m"
BATTERY_V = "/drone/battery_v"
POSITION_NED = "/drone/position_ned"  # (x, y, z) tuple
HEADING_DEG = "/drone/heading_deg"
ARMED = "/drone/armed"

SURVEY_WAYPOINTS = "/mission/waypoints"
CURRENT_WP_IDX = "/mission/current_wp_idx"
DETECTIONS = "/mission/detections"  # list of dicts
SURVEY_COMPLETE = "/mission/survey_complete"

TRACKER_STATE = "/landing/tracker_state"  # raw /api/state dict
ALIGN_CENTERED = "/landing/aligned"
RANSAC_OK = "/landing/ransac_ok"
ARUCO_VISIBLE = "/landing/aruco_visible"
ARUCO_X_M = "/landing/aruco_x_m"
ARUCO_Y_M = "/landing/aruco_y_m"
ARUCO_Z_M = "/landing/aruco_z_m"

CHARGE_START_SOC = "/charging/start_soc"
CHARGE_END_SOC = "/charging/end_soc"
