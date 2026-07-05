# stargazing-core

Shared astronomical computation package for the stargazing toolchain —
telescope optics, deep-sky catalog, and equipment presets used by both
[mcp-stargazing](https://github.com/StarGazer1995/mcp-stargazing) and
[stargazing-place-finder](https://github.com/StarGazer1995/stargazing-place-finder).

## Installation

```bash
pip install stargazing-core
```

## Quick Start

```python
from stargazing_core import TelescopeConfig, TelescopeOptics, TELESCOPE_PRESETS

# Look up a preset
config = TELESCOPE_PRESETS['seestar-s50']

# Compute derived optics
optics = config.compute_optics()
print(f'FOV: {optics.fov_width_deg:.1f}° × {optics.fov_height_deg:.1f}°')
# → FOV: 1.7° × 1.3°
```

## Development

```bash
uv sync
uv run pytest -v
uv run ruff format --check . && uv run ruff check .
```
