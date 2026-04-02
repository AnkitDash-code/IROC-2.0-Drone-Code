#!/bin/bash
# Network Disconnect Monitor
# Continuously checks network connectivity and dumps routing/interface debug logs when a drop occurs.

LOG_DIR="$(dirname "$(realpath "$0")")/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/network_drop_debug.log"

TARGET="8.8.8.8" # IP to check global connectivity (Google DNS)

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Network monitor started. Watching for disconnects..." | tee -a "$LOG_FILE"

was_connected=1

while true; do
    # Try to ping the target
    if ping -c 1 -W 2 "$TARGET" >/dev/null 2>&1; then
        if [ "$was_connected" -eq 0 ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Network RECONNECTED." | tee -a "$LOG_FILE"
            was_connected=1
        fi
    else
        # If ping fails but we were previously connected, trigger a log dump
        if [ "$was_connected" -eq 1 ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] DISCONNECT DETECTED! Grabbing system logs..." | tee -a "$LOG_FILE"
            
            {
                echo "================= DISCONNECT DEBUG DUMP =================="
                echo "TIMESTAMP: $(date)"
                echo ""
                echo "--- 1. ACTIVE IP ROUTES (ip route) ---"
                ip route
                echo ""
                echo "--- 2. INTERFACE STATUS (ip addr) ---"
                ip addr
                echo ""
                echo "--- 3. NETWORK MANAGER DEVICE STATES ---"
                nmcli device status
                echo ""
                echo "--- 4. ACTIVE CONNECTIONS ---"
                nmcli connection show --active
                echo ""
                echo "--- 5. KERNEL LOGS (dmesg - HW/Driver drops) ---"
                dmesg | tail -n 30
                echo ""
                echo "--- 6. NETWORK MANAGER LOGS (Software reconnect attempts) ---"
                journalctl -u NetworkManager --since "5 minutes ago" --no-pager | tail -n 40
                echo "=========================================================="
                echo ""
            } >> "$LOG_FILE"
            
            was_connected=0
        fi
    fi
    sleep 3
done
