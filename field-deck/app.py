"""Minimal static server for the field deck (Databricks Apps).

Serves this directory (including deck-assets/) and maps "/" to the deck.
Uses only the Python standard library, so no requirements.txt is needed.
"""
import http.server
import os
import socketserver

PORT = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
ROOT = os.path.dirname(os.path.abspath(__file__))
DECK = "/ucode-field-deck.html"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_GET(self):
        if self.path in ("/", ""):
            self.path = DECK
        return super().do_GET()


if __name__ == "__main__":
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"serving {ROOT} on 0.0.0.0:{PORT}")
        httpd.serve_forever()
