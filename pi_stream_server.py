#!/usr/bin/env python3
import atexit
import os
import shutil
import signal
import subprocess
from flask import Flask, send_from_directory


HLS_DIR = os.environ.get("PI_HLS_DIR", "/tmp/pi_hls")
HTTP_HOST = os.environ.get("PI_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("PI_HTTP_PORT", "8081"))

WIDTH = int(os.environ.get("PI_WIDTH", "1280"))
HEIGHT = int(os.environ.get("PI_HEIGHT", "720"))
FPS = int(os.environ.get("PI_FPS", "60"))
BITRATE = int(os.environ.get("PI_BITRATE", "6000000"))
INTRA = int(os.environ.get("PI_INTRA", "30"))

app = Flask(__name__)
_stream_proc = None


def start_streamer() -> subprocess.Popen:
    os.makedirs(HLS_DIR, exist_ok=True)

    for name in os.listdir(HLS_DIR):
        path = os.path.join(HLS_DIR, name)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    cmd = (
        f"rpicam-vid --nopreview -t 0 --width {WIDTH} --height {HEIGHT} "
        f"--framerate {FPS} --bitrate {BITRATE} --intra {INTRA} "
        "--codec h264 --inline -o - | "
        "gst-launch-1.0 -q fdsrc ! h264parse config-interval=1 ! mpegtsmux ! "
        f"hlssink target-duration=1 max-files=8 "
        f"playlist-location={HLS_DIR}/stream.m3u8 "
        f"location={HLS_DIR}/segment_%05d.ts"
    )

    return subprocess.Popen(["bash", "-lc", cmd])


def stop_streamer() -> None:
    global _stream_proc
    if _stream_proc and _stream_proc.poll() is None:
        _stream_proc.terminate()
        try:
            _stream_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _stream_proc.kill()


@app.route("/")
def root():
    return "Pi HLS server is running. Use /hls/stream.m3u8\n"


@app.route("/health")
def health():
    ok = _stream_proc is not None and _stream_proc.poll() is None
    return {"streamer_running": ok, "hls_dir": HLS_DIR}, (200 if ok else 500)


@app.route("/hls/<path:name>")
def hls(name):
    if name.endswith(".m3u8"):
        mime = "application/vnd.apple.mpegurl"
    elif name.endswith(".ts"):
        mime = "video/mp2t"
    else:
        mime = None

    resp = send_from_directory(HLS_DIR, name, mimetype=mime)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


def _handle_signal(_sig, _frame):
    stop_streamer()
    raise SystemExit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    atexit.register(stop_streamer)

    _stream_proc = start_streamer()
    print(f"[*] Pi HLS server on http://{HTTP_HOST}:{HTTP_PORT}")
    print(f"[*] HLS URL: http://{HTTP_HOST}:{HTTP_PORT}/hls/stream.m3u8")
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False)
