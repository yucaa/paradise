"""Copy the live piece into a checkout of the site repo (github.com/yucaa/paradise,
a Next.js shell deployed by Vercel), which serves it as static files from
public/live/. The pipeline's source goes alongside, in pipeline/.

    python viewer/publish.py ../paradise-site
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "viewer" / "live"
DATA = ["meadow", "constellation.json", "heaven.png"]           # what the page loads
PIPELINE = ["paradise", "blender", "pyproject.toml", "uv.lock", ".python-version", "viewer/serve.py",
            "viewer/publish.py"]


def copy(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main(site: Path):
    public = site / "public"
    shutil.rmtree(public / "live", ignore_errors=True)
    for f in LIVE.iterdir():
        if f.is_file():
            copy(f, public / "live" / f.name)
    for d in DATA:
        copy(LIVE / "data" / d, public / "live" / "data" / d)
    copy(ROOT / "viewer" / "style.css", public / "style.css")      # live/index.html loads ../style.css
    shutil.rmtree(site / "pipeline", ignore_errors=True)
    for p in PIPELINE:
        copy(ROOT / p, site / "pipeline" / p)
    print("published to", site)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
