import threading, time, http.server, os, signal, sys

PORT = int(os.environ.get("PORT", 8080))


class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, *a):
        pass


def run_health_server():
    s = http.server.HTTPServer(("0.0.0.0", PORT), HealthHandler)
    s.serve_forever()


t = threading.Thread(target=run_health_server, daemon=True)
t.start()

from main import main
main()
