with open("guarded_mission.py", "r") as f:
    text = f.read()

# Make parser accept extra args gracefully
text = text.replace("return parser.parse_args()", "args, _ = parser.parse_known_args()\n    return args")

# Add generic exception fallback catch in main
old_main_except = """    except ManualOverride as exc:
        print(f"[MANUAL] {exc}")
        print("[MANUAL] Automation stopped. Pilot has priority control.")
        return 2"""

new_main_except = """    except ManualOverride as exc:
        print(f"[MANUAL] {exc}")
        print("[MANUAL] Automation stopped. Pilot has priority control.")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unhandled exception occurred: {exc}")
        print("[SAFETY] Attempting to land immediately...")
        try:
            if mission.master and mission.motors_armed():
                mission.set_mode("LAND")
                print("[SAFETY] Switched to LAND mode. Waiting to disarm...")
                import drone_control as dc
                dc.master.motors_disarmed_wait()
                print("[SAFETY] Successfully landed and disarmed after error.")
        except Exception as land_exc:
            print(f"[FATAL] Failsafe landing after exception failed: {land_exc}")
        return 99"""
if old_main_except in text:
    text = text.replace(old_main_except, new_main_except)

with open("guarded_mission.py", "w") as f:
    f.write(text)
print("Main patch applied!")
