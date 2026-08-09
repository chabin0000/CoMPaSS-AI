param([string]$Root = (Split-Path -Parent $PSScriptRoot))
uv run python -m compass_ai mock-e2e --root $Root
