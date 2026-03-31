import subprocess
import time

# --- CONFIGURATION ---
# The IP address of your laptop on the Wi-Fi network
PC_IP = "10.152.54.50" 
PORT = "5000"

def start_stream_relay():
    """Starts the GStreamer pipeline to bounce the Pi's stream to the PC."""
    
    print(f"[*] Initializing Drone Video Relay...")
    print(f"[*] Catching stream from Raspberry Pi on port {PORT}")
    print(f"[*] Bouncing stream to PC at {PC_IP}:{PORT}")

    # The exact low-latency GStreamer relay command
    gstreamer_cmd = [
        "gst-launch-1.0", "-v",
        "udpsrc", f"port={PORT}",
        "!", "udpsink", f"host={PC_IP}", f"port={PORT}"
    ]

    try:
        # subprocess.run will block and keep the stream open until you press Ctrl+C
        subprocess.run(gstreamer_cmd)
    except KeyboardInterrupt:
        print("\n[*] Video relay stopped by user.")
    except Exception as e:
        print(f"\n[!] An error occurred: {e}")

if __name__ == "__main__":
    # Optional: Add a small delay if running on boot to ensure networks are up
    time.sleep(2)
    start_stream_relay()
