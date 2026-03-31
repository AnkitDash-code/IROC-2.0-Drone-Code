#!/bin/bash
set -euo pipefail
LOG=/tmp/jetson-network-fix.log
{
  echo "$(date +'%F %T') [jetson-network-fix] starting"
  # Apply persistent nmcli connection settings (no sudo required when run as root)
  nmcli connection modify direct-eth ipv4.never-default yes
  nmcli connection modify direct-eth ipv4.ignore-auto-routes yes
  nmcli connection modify direct-eth ipv4.ignore-auto-dns yes

  # Restart the connection and NetworkManager to rebuild routing table
  nmcli connection up direct-eth || echo "nmcli connection up failed" >&2
  sleep 2
  systemctl restart NetworkManager || echo "systemctl restart NetworkManager failed" >&2
  echo "$(date +'%F %T') [jetson-network-fix] finished"
} >> "$LOG" 2>&1
