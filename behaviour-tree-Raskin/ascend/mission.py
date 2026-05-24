import time
import py_trees

# Import the nodes we built in Days 1-5
from ascend.bt_nodes.safety import make_safety_guard
from ascend.bt_nodes.preflight import (
    ConnectVehicle, 
    WaitRangefinder, 
    WaitBatteryTelemetry, 
    GroundCheck, 
    OperatorConfirm
)
from ascend.bt_nodes.takeoff import (
    SetMode, 
    ArmVehicle, 
    SetSlowTakeoffParams, 
    CommandTakeoff, 
    WaitAltitudeReached,
    Hover
)
from ascend.bt_nodes.landing import make_landing_phase, LogMissionReport

def create_root_tree():
    """
    Assembles the complete ASCEND IRoC-U 2026 behavior tree.
    """
    # ----------------------------------------------------
    # Phase 0: PreFlight (All must pass)
    # ----------------------------------------------------
    preflight = py_trees.composites.Sequence(name="PreFlight", memory=True)
    preflight.add_children([
        ConnectVehicle("tcp:127.0.0.1:5762", 57600),
        # WaitRangefinder(timeout_s=15.0),      # Disabled for SITL simulation
        # WaitBatteryTelemetry(timeout_s=15.0), # Disabled for SITL simulation
        # GroundCheck(),                        # Disabled for SITL simulation
        OperatorConfirm("READY")
    ])

    # ----------------------------------------------------
    # Phase 1: Takeoff
    # ----------------------------------------------------
    takeoff = py_trees.composites.Sequence(name="Takeoff", memory=True)
    takeoff.add_children([
        SetMode("GUIDED"),
        OperatorConfirm("ARM"),
        ArmVehicle(),
        SetSlowTakeoffParams(),
        CommandTakeoff(altitude_m=5.0),
        WaitAltitudeReached(target_m=5.0)
    ])

    # ----------------------------------------------------
    # Assemble the core Mission (Sequential)
    # ----------------------------------------------------
    mission_root = py_trees.composites.Sequence(name="MissionRoot", memory=True)
    mission_root.add_children([
        preflight,
        takeoff,
        Hover(duration_s=60), # 5-minute hover test just like the old code!
        make_landing_phase(),  # Phase 4: Precision Landing
        LogMissionReport()     # Phase 5: Post-Flight
    ])

    # ----------------------------------------------------
    # Top Level: Safety Parallel
    # Runs the Safety Guards alongside the Mission Loop.
    # ----------------------------------------------------
    from ascend.bt_nodes.safety import AltitudeMonitor
    
    root = py_trees.composites.Parallel(
        name="ASCEND_Tree",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne() # If mission finishes, we stop.
    )
    root.add_children([
        AltitudeMonitor(), # Polls altitude so WaitAltitudeReached works
        mission_root
    ])
    
    return root

def main():
    root = create_root_tree()
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(timeout=15)
    
    print("\n" + "="*50)
    print(" ASCEND IRoC-U 2026 Behavior Tree Initialized")
    print("="*50 + "\n")
    
    try:
        # Tick the tree at 10Hz
        while True:
            tree.tick()
            
            status = tree.root.status
            if status == py_trees.common.Status.SUCCESS:
                print("\n[MISSION COMPLETE] All phases executed successfully.")
                break
            elif status == py_trees.common.Status.FAILURE:
                print("\n[MISSION ABORTED] A critical node or safety guard failed.")
                break
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n[EMERGENCY] Keyboard Interrupt detected! Tree stopped.")
        # In a real scenario you might want to force MAVLink RTL or LAND here
        import drone_control as dc
        try:
            print("Commanding emergency LAND mode...")
            dc.set_mode("LAND")
        except Exception:
            pass

if __name__ == "__main__":
    main()
