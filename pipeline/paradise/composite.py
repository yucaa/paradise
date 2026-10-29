"""Title card, then the rendered film with the overlay on top, encoded to MP4."""
import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

from .connectome import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default=str(ROOT / "out" / "film"))
    ap.add_argument("--hud", default=str(ROOT / "out" / "hud"))
    ap.add_argument("--titles", default=str(ROOT / "out" / "titles"), help="frames prepended as-is")
    ap.add_argument("--out", default=str(ROOT / "out" / "paradise.mp4"))
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()
    frames = sorted(Path(args.film).glob("*.png"))
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for p in sorted(Path(args.titles).glob("*.png")) if args.titles else []:
        proc.stdin.write(Image.open(p).convert("RGB").tobytes())
    for p in frames:
        bg = Image.open(p).convert("RGBA")
        hud = Path(args.hud) / p.name
        if hud.exists():
            bg = Image.alpha_composite(bg, Image.open(hud))
        proc.stdin.write(bg.convert("RGB").tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", args.out, len(frames), "frames")


if __name__ == "__main__":
    main()
