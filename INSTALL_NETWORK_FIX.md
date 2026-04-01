Install Network Fix

Follow these steps on the Jetson as root (or with sudo):

1. Copy the persistent connection into NetworkManager's system folder and set correct permissions:

```bash
sudo cp system-connections/direct-eth.nmconnection /etc/NetworkManager/system-connections/
sudo chown root:root /etc/NetworkManager/system-connections/direct-eth.nmconnection
sudo chmod 600 /etc/NetworkManager/system-connections/direct-eth.nmconnection
```

2. Reload NetworkManager so it reads the new connection profile:

```bash
sudo systemctl restart NetworkManager
```

3. Verify the connection settings are applied and the `direct-eth` connection uses `never-default` / `ignore-auto-*`:

```bash
nmcli -g ipv4.never-default,ipv4.ignore-auto-routes,ipv4.ignore-auto-dns connection show direct-eth
nmcli connection show direct-eth
ip addr show dev eth0
ip route
```

4. Optional: keep the helper service (already present) but it is no longer required.

If Jetson still prefers `wlan0` for default route after this change, check that `direct-eth`'s `never-default` and `ignore-auto-routes` are set and that no other connection has higher autoconnect priority.
