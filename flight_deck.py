from flask import Flask, Response, jsonify
import urllib.request
import urllib.error
import os

app = Flask(__name__)
PI_WS_URL = os.environ.get("PI_WS_URL", "auto")
WS_PUBLIC_PORT = os.environ.get("WS_PUBLIC_PORT", "8082")
PI_HTTP_BASE = os.environ.get("PI_HTTP_BASE", "http://10.55.0.2:8082")

@app.route("/")
def index():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Jetson Flight Deck</title>
  <style>
    body { background:#111; color:#eee; font-family:sans-serif; margin:0; padding:16px; }
    h3 { margin:0 0 12px 0; }
    #v { width:90vw; max-width:1100px; border:1px solid #333; border-radius:10px; background:#000; display:block; }
    #status { margin-top:10px; color:#9ad; font-size:14px; }
    .err { color:#ff8080; }
  </style>
</head>
<body>
  <h3>Jetson Web Viewer (Pi WebSocket Feed)</h3>
  <img id="v" alt="Live stream" />
  <div id="status">Loading...</div>

  <script>
    const img = document.getElementById('v');
    const statusEl = document.getElementById('status');
    const wsUrl = '__PI_WS_URL__';
    const wsPublicPort = '__WS_PUBLIC_PORT__';
    let lastUrl = null;

    function setStatus(msg, err=false) {
      statusEl.textContent = msg;
      statusEl.className = err ? 'err' : '';
      console.log(msg);
    }

    function resolvedWsUrl() {
      if (wsUrl === 'auto') {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        return `${proto}://${location.hostname}:${wsPublicPort}/ws`;
      }
      return wsUrl;
    }

    function connect() {
      const target = resolvedWsUrl();
      setStatus('Connecting: ' + target);
      const ws = new WebSocket(target);
      ws.binaryType = 'blob';

      ws.onopen = () => setStatus('Connected');
      ws.onerror = () => setStatus('WebSocket error', true);
      ws.onclose = () => {
        setStatus('Disconnected, retrying in 1s...', true);
        setTimeout(connect, 1000);
      };

      ws.onmessage = (ev) => {
        const url = URL.createObjectURL(ev.data);
        img.src = url;
        if (lastUrl) {
          URL.revokeObjectURL(lastUrl);
        }
        lastUrl = url;
      };
    }

    connect();
  </script>
</body>
</html>
"""
    html = html.replace("__PI_WS_URL__", PI_WS_URL)
    html = html.replace("__WS_PUBLIC_PORT__", WS_PUBLIC_PORT)
    return Response(html, mimetype="text/html")

@app.route("/health")
def health():
  src_url = f"{PI_HTTP_BASE}/health"
  try:
    with urllib.request.urlopen(src_url, timeout=5) as upstream:
      payload = upstream.read()
      status = upstream.getcode()
  except urllib.error.HTTPError as err:
    return jsonify({"ready": False, "error": f"upstream http {err.code}", "upstream": src_url}), err.code
  except Exception as err:
    return jsonify({"ready": False, "error": f"upstream unreachable: {err}", "upstream": src_url}), 502

  return Response(payload, status=status, mimetype="application/json")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)