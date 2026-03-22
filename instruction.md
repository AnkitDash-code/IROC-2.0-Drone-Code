# guarded_mission Instructions

This document explains how to run and operate `guarded_mission.py` safely.

## What This Script Does

`guarded_mission.py` executes a manual-gated safety mission:
1. Connect to flight controller
2. Confirm RANGEFINDER altitude stream is valid
3. Verify ground altitude
4. Set LOITER
5. Arm
6. Take off slowly to selected altitude
7. Start 5-minute hover countdown only after altitude is reached
8. LAND
9. Disarm confirmation/check

Pilot RC stick input has priority and can terminate autonomy.

## Integration With Existing Code

The script now reuses core control calls from `drone_control.py`:
- `connect_to_vehicle()` for connection
- `set_mode()` for mode changes
- `arm_vehicle()` for arming
- `takeoff()` for takeoff command

Additional safety gating (rangefinder checks, override detection, mission state logic) remains inside `guarded_mission.py`.

## Commands

Normal:

```bash
python3 guarded_mission.py --connect /dev/ttyACM0 --baud 57600 --hover-seconds 300
```

No-propeller mode:

```bash
python3 guarded_mission.py --connect /dev/ttyACM0 --baud 57600 --hover-seconds 300 --no-propeller-mode
```

Arguments:
- `--connect`: MAVLink connection string (default `/dev/ttyACM0`)
- `--baud`: serial baud rate (default `57600`)
- `--hover-seconds`: hover duration in seconds (default `300`)
- `--no-propeller-mode`: enables motor output cap mode (30%) after explicit confirmation

## Altitude Source

Altitude is read from `RANGEFINDER` only.

`GLOBAL_POSITION_INT` altitude is not used for mission safety decisions in this script.

## Operator Prompts

You will be prompted for:
- Target altitude (0.5 m to 3.0 m)
- Confirmation tokens by phase: `READY`, `LOITER`, `ARM`, `TAKEOFF`, `HOVER`, `LAND`
- If no-propeller mode is enabled: `NOPROP`

Tokens must match exactly. After `LAND` confirmation, the vehicle lands automatically and auto-disarms (no final confirmation needed).

## Mission Flow Details

### 1) Preflight
- Wait for heartbeat
- Require live rangefinder data
- Require stable near-ground altitude

### 2) No-Propeller Mode (Optional)
- If enabled, applies `MOT_SPIN_MAX = 0.30`
- Original value is saved and restored on exit

### 3) LOITER + Arm
- Script sets LOITER before takeoff
- Arming is performed only after manual confirmation

### 4) Slow Takeoff
- Temporarily sets:
  - `WPNAV_SPEED_UP`
  - `WPNAV_ACCEL_Z`
- Sends takeoff command (via `drone_control.takeoff()`)
- Confirms altitude reached at about 95% of target
- Restores original parameter values in `finally` block

### 5) Hover Countdown
- Countdown begins only after takeoff altitude reach check passes
- Prints timer as `mm:ss`

### 6) Landing & Auto-Disarm
- Calls `drone_control.land_vehicle()` for proper landing sequence
- Internally uses `master.motors_disarmed_wait()` to wait for auto-disarm
- Vehicle auto-disarms when FC detects ground altitude (≤ 0.15m)
- No manual disarm confirmation required
- Mission completes after land and disarm

## Safety Behavior

### Altitude Limits
- Above 3.0 m: force LAND
- Above 4.0 m: force LAND with critical warning

### RC Override Priority
- RC channels are monitored continuously
- Large stick movement around neutral triggers `ManualOverride`
- Script exits autonomous sequence and returns control to pilot

### Keyboard Interrupt
- `Ctrl+C` while airborne: immediately lands and waits for auto-disarm
- `Ctrl+C` on/near ground: immediately disarms if armed

### Forced LAND Mission End
- If forced LAND occurs during takeoff or hover, script ends mission after landing/disarm wait

## Exit Codes

- `0`: mission completed
- `1`: runtime error path
- `2`: manual RC override detected (autonomy stopped)
- `130`: keyboard interrupt

## Auto-Disarm Behavior

When the vehicle lands, the flight controller automatically disarms based on altitude thresholds (typically ≤ 0.15m continuously).

The `drone_control.land_vehicle()` function blocks on `master.motors_disarmed_wait()` until this auto-disarm occurs, ensuring the script waits for true mission completion.

## Recommended Pre-Run Checklist

- Verify serial path and baud
- Confirm rangefinder is active
- Confirm RC takeover works
- Validate flight-controller failsafes/geofence/battery limits
- Start with props-off bench testing or SITL

## Troubleshooting

### No heartbeat
- Verify telemetry/USB link
- Verify `--connect` and `--baud`

### No rangefinder data
- Check rangefinder wiring and FC configuration
- Confirm RANGEFINDER messages are present

### Mode change fails
- Ensure LOITER and LAND modes exist on your firmware and vehicle type

### Parameter set not confirmed
- Some firmware builds may not expose `WPNAV_*` or `MOT_SPIN_MAX`
- Check terminal logs for read/set confirmation messages

### Manual override triggers too easily
- RC sticks may not center exactly at 1500
- Tune `RC_INPUT_DEVIATION` in `guarded_mission.py`
