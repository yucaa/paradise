"""Serve the live brain to the world from this machine.

Opens a Cloudflare quick tunnel (no account needed) to the brain server on
ws://127.0.0.1:8766, writes its public address to public/live/brain.json in a
checkout of the site repo, and pushes it, so the deployed page finds the brain.
Run alongside `python -m paradise.live serve`:

    python -m paradise.tunnel ../paradise-site

A quick tunnel gets a new address each time it starts; the push (and Vercel's
redeploy, about a minute) follows it.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .live import PORT

# On PATH, or the standalone release binary in %LOCALAPPDATA%\cloudflared.
CLOUDFLARED = shutil.which("cloudflared") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "cloudflared" / "cloudflared.exe")


def publish(site: Path, url: str):
    path = site / "public" / "live" / "brain.json"
    path.write_text(json.dumps({"url": url}) + "\n", encoding="utf-8")
    git = lambda *a: subprocess.run(["git", "-C", str(site), *a], check=True)
    git("add", str(path))
    if subprocess.run(["git", "-C", str(site), "diff", "--cached", "--quiet"]).returncode:
        git("commit", "-q", "-m", f"Brain served at {url}")
        git("push", "-q", "origin", "HEAD")
    print("published", url, flush=True)


def main(site: Path):
    p = subprocess.Popen([CLOUDFLARED, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = None
    for line in p.stdout:
        if url is None and (m := re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)):
            url = m.group(0).replace("https://", "wss://")
            publish(site, url)
    sys.exit(p.wait())


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
