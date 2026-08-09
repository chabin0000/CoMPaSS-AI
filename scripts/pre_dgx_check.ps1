param([string]$Root = (Split-Path -Parent $PSScriptRoot))
uv run python -m compass_ai pre-dgx-check --root $Root
