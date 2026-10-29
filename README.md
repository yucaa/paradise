# Paradise

A fruit fly's heaven, extracted from its connectome. The visual world that
most pleases one *Drosophila* brain (FlyWire 783, simulated) paints a meadow;
a fly lives in it, and its brain is shown in the margins as it looks around.

- `public/live/`: the piece, a static three.js page (the site's `/` redirects here).
- `pipeline/`: the Python and Blender source that simulates the brain, extracts
  heaven and builds the meadow.

## The brain

The page's brain readout comes from a GPU server you run yourself:

```bash
cd pipeline
uv run python -m paradise.live serve      # ws://127.0.0.1:8766
```

The page connects to it at `ws://127.0.0.1:8766`, or to `?brain=wss://...`.
Without it the meadow and the fly still run, and the brain panels show offline.

## Updating

From the working repo: `python viewer/publish.py <path to this checkout>`, then commit and push.
