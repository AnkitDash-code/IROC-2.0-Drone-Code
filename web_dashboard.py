from optical_flow_realsense.web_dashboard import *

if __name__ == "__main__":
    from optical_flow_realsense.web_dashboard import app
    print("Web Dashboard starting...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
