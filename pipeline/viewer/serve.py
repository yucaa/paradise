"""Serve the repo for the viewers on http://127.0.0.1:8765, telling the
browser never to reuse a stale copy, so edits show on a plain refresh."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ThreadingHTTPServer(("127.0.0.1", 8765), partial(NoCache, directory=str(root))).serve_forever()
