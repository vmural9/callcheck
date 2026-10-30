Throwaway scripts, not part of the package, excluded from lint and tests.
Findings go to docs/research.
Run with `uv run --extra bedrock python spikes/<name>.py`.
`bargein` expects `hears` to have run first: it reads `spikes/out/hello16k.pcm` (written by `hears`) and otherwise sends a tone.
