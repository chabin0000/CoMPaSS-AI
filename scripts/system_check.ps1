param([string]$Root = (Split-Path -Parent $PSScriptRoot))
uv run python -m compass_ai system-check --root $Root
