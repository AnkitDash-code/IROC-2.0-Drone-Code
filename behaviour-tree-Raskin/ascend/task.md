# ASCEND BT Architecture — File Tracking & Task List

This document tracks which components from the old monolithic codebase (`guarded_mission.py` and `drone_control.py`) have been successfully ported over to the new Behavior Tree nodes in the `ascend/` directory.

- `[x]` **Day 1: Foundation**
  - **Created:** `ascend/blackboard_keys.py`, `ascend/bt_nodes/__init__.py`, `ascend/bt_nodes/base.py`
  - **Source Code Mapped:** General BT architectural structure (No old code ported yet).

- `[x]` **Day 2: Safety Guard**
  - **Created:** `ascend/bt_nodes/safety.py` (`BatteryMonitor`, `AltitudeMonitor`)
  - **Source Code Mapped:** 
    - `guarded_mission.py` -> `_extract_pack_voltage_v()`
    - `guarded_mission.py` -> `get_altitude_m()`
    - `guarded_mission.py` -> `enforce_battery_failsafe()`

- `[x]` **Day 3: Pre-flight & Takeoff Nodes**
  - **Created:** `ascend/bt_nodes/preflight.py` (`ConnectVehicle`, `WaitRangefinder`, `WaitBatteryTelemetry`, `GroundCheck`, `OperatorConfirm`)
  - **Created:** `ascend/bt_nodes/takeoff.py` (`SetMode`, `ArmVehicle`, `SetSlowTakeoffParams`, `CommandTakeoff`, `WaitAltitudeReached`)
  - **Source Code Mapped:** 
    - `guarded_mission.py` -> `confirm_step()`
    - `guarded_mission.py` -> `wait_for_rangefinder()`
    - `guarded_mission.py` -> `wait_for_ground()`
    - `guarded_mission.py` -> `set_slow_takeoff_profile()`
    - `guarded_mission.py` -> `request_rangefinder_stream()`, `request_battery_stream()`
    - `drone_control.py` -> `arm_vehicle()`, `set_mode()`, `takeoff()` (without sleep)

- `[x]` **Day 4: Survey Phase**
  - **Created:** `ascend/bt_nodes/survey.py`
  - **Source Code Mapped:** `drone_control.py` -> `rotate_towards()`, `move_body_ned()`

- `[x]` **Day 5: Landing Phase**
  - **Created:** `ascend/bt_nodes/landing.py`
  - **Source Code Mapped:** 
    - `guarded_mission.py` -> `descend_to_base_hold_altitude()`
    - `guarded_mission.py` -> `align_over_base_station()` (ArUco Math completely ported)
    - `drone_control.py` -> `special_landing()`

- `[x]` **Day 6: Root Tree Assembly & Return to Base**
  - **Created:** `ascend/mission.py`
  - **Source Code Mapped:** `drone_control.py` (RTB movement logic replaced by `NavigateHome()`)

- `[x]` **Day 7: SITL Full Run**
  - **Status:** Full mission completed in SITL! Armed, took off to 3m, flew 6-waypoint lawnmower survey, all phases executed successfully.

---

## 🚁 Before Flying the Real Drone — Migration Checklist

> Everything below was changed or disabled for SITL. Do ALL of these before the first real flight.

---

### 📝 Code Changes

#### [`ascend/mission.py`](file:///d:/college projects/IROC-U/IROC-2.0-Drone-Code-main/IROC-2.0-Drone-Code-main/ascend/mission.py)
- `[ ]` Uncomment `WaitRangefinder(timeout_s=15.0)` in PreFlight block
- `[ ]` Uncomment `WaitBatteryTelemetry(timeout_s=15.0)` in PreFlight block
- `[ ]` Uncomment `GroundCheck()` in PreFlight block
- `[x]` Replace `AltitudeMonitor()` with `make_safety_guard()` in the root Parallel node
- `[ ]` Change `SetMode("GUIDED")` → `SetMode("LOITER")` in Takeoff phase
- `[ ]` *(For full mission only)* Replace `Hover(duration_s=60)` with `make_survey_phase()` and `NavigateHome()`
- `[ ]` *(For full mission only)* Import those functions from `ascend.bt_nodes.survey` at the top of the file
- `[ ]` Change connection string `"tcp:127.0.0.1:5762"` → `"/dev/ttyACM0"` (check Jetson port with `ls /dev/ttyACM*`)

#### [`ascend/bt_nodes/safety.py`](file:///d:/college projects/IROC-U/IROC-2.0-Drone-Code-main/IROC-2.0-Drone-Code-main/ascend/bt_nodes/safety.py)
- `[ ]` Confirm `BATT_LOW_V = 16.0` (4S × 4.0V) is correct for your pack
- `[ ]` Confirm `BATT_CRIT_V = 15.2` (4S × 3.8V) is correct for your pack
- `[ ]` Confirm `ALT_MAX_M = 6.5` matches the IRoC-U competition ceiling

#### [`ascend/bt_nodes/landing.py`](file:///d:/college projects/IROC-U/IROC-2.0-Drone-Code-main/IROC-2.0-Drone-Code-main/ascend/bt_nodes/landing.py)
- `[ ]` Wire up OAK-D camera tracker server so `ArUcoAlign` stops hitting SITL fallback
- `[ ]` Confirm `TRACKER_URL = "http://localhost:5000/api/state"` matches where tracker runs on Jetson
- `[ ]` Confirm `HOLD_ALT_M = 1.5` is the right descent height for your base station marker

#### [`ascend/bt_nodes/survey.py`](file:///d:/college projects/IROC-U/IROC-2.0-Drone-Code-main/IROC-2.0-Drone-Code-main/ascend/bt_nodes/survey.py)
- `[ ]` Wire up `YOLODetect` stub → real OAK-D YOLO pipeline
- `[ ]` Wire up `VLMVerify` stub → real Moondream 2.5 call
- `[ ]` Confirm `ARENA_X_M = 10.67` and `ARENA_Y_M = 7.62` match the real IRoC-U field size
- `[ ]` Confirm `LANE_SPACING = 1.5` and `SURVEY_SPEED = 0.5` are safe for real flight

---

### 🖥️ Mission Planner Parameter Changes

> Open **CONFIG → Full Parameter List**, change each value, then click **Write Params**

| Parameter | SITL Value | Real Drone Value | Why |
|---|---|---|---|
| `ARMING_SKIPCHK` | `-1` | `0` | Re-enable all pre-arm safety checks |
| `RNGFND1_TYPE` | `0` (disabled) | Your sensor type (e.g. `2` for MaxBotix) | Re-enable rangefinder for altitude |
| `BATT_LOW_VOLT` | *(any)* | `16.0` | Low battery RTL threshold (4S) |
| `BATT_CRT_VOLT` | *(any)* | `15.2` | Critical battery LAND threshold (4S) |
| `FS_BATT_ENABLE` | *(any)* | `2` (Land) | Enable battery failsafe |
| `FS_THR_ENABLE` | *(any)* | `1` | Enable RC loss failsafe |

---

### 🤖 Jetson Hardware Setup (First Real Flight)

- `[ ]` Confirm USB serial port: run `ls /dev/ttyACM*` and update connection string in `mission.py`
- `[ ]` Start ArUco tracker server **before** running mission: `python tracker_server.py`
- `[ ]` Confirm OAK-D camera is detected: `python -c "import depthai; print('OAK-D OK')"`
- `[ ]` Do a dry-run without props: arm, confirm telemetry flows, Ctrl+C to abort
- `[ ]` Check that `detections.jsonl` log path exists: `/home/jetson123/Drone/detections.jsonl`
