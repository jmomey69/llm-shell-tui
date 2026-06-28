# llm-shell

A terminal UI control panel for [llama.cpp](https://github.com/ggerganov/llama.cpp) — manage your local inference server, browse models, quantize, and convert, all from one keyboard-driven interface.

Built with [Textual](https://textual.textualize.io/).

## Tabs

<!-- AUTO:tabs -->
| Tab | Sections | Buttons | Description |
|-----|----------|---------|-------------|
| **Server** | `Live logs (last 80 lines):` | **Start**, **Stop**, **Restart**, **Enable auto-start**, **Disable auto-start** | Start/stop/restart the `llama-server` systemd user service; tail live logs; edit port, context size, and GPU/CPU layer counts before restarting |
| **Model Finder** | — | **Refresh**, **Open on HuggingFace ↗**, **Download (hf-cli)**, **Copy ollama pull cmd** | Browse GGUF models on HuggingFace filtered by VRAM budget, open on HF, or copy an `ollama pull` command |
| **Models** | `GGUF models in ~/` | **Refresh**, **Launch server with selected**, **Delete selected** | List every `.gguf` in `~/`, show size and estimated VRAM fit, launch directly or delete |
| **Run (direct)** | `Launch server directly (foreground, MTP speculative decoding)`, `Server output:` | **Start**, **Stop**, **Auto Config** | Launch `llama-server` in the foreground without systemd; **Auto Config** reads model metadata and your GPU's free VRAM to suggest optimal GPU/CPU layer split and context length |
| **Convert / Quantize** | `Quantize a GGUF model`, `Convert HuggingFace model → GGUF`, `Output:` | **Quantize**, **Convert to GGUF** | Quantize any GGUF to a smaller quant type; convert a HuggingFace checkpoint to GGUF |
<!-- /AUTO:tabs -->

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (or any env with `textual` installed)
- A built or installed `llama-server` binary
- Linux with systemd (for the Server tab); the Run tab works without it

## Usage

```bash
uv run --with textual llm-shell.py
```

Or add an alias to `~/.bashrc`:

```bash
alias llm-shell='uv run --with textual ~/llm-shell.py'
```

## Supported quant types

<!-- AUTO:quants -->
`Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`, `F16`, `F32`
<!-- /AUTO:quants -->

## Auto Config

The **Auto Config** button in the Run tab reads the selected GGUF file's binary metadata (block count, embedding size, KV head count) and your GPU's free VRAM to calculate:

- How many layers to offload to GPU vs CPU
- A context length that fits in the remaining VRAM after the model weights

It reserves at least 1 GB for KV cache before fitting GPU layers, so the suggested values actually start without OOM errors.

## Systemd service

`systemd/llama-server.service` is a ready-to-use user service. Install it:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/llama-server.service ~/.config/systemd/user/
# edit ExecStart to point at your binary and model
systemctl --user daemon-reload
systemctl --user enable --now llama-server
```

The included service is tuned for an **RTX 5060 Ti (16 GB)** with a 15.68 GB Q4\_K\_M model:
`-ngl 55 -c 16384` (55 GPU layers, 16 K context, no MTP to keep within VRAM budget).

## Configuration

Edit the constants at the top of `llm-shell.py`:

<!-- AUTO:config -->
```python
BUILD_BIN = Path.home() / "Documents/ai-assistant/ai-quant-platform/llama.cpp/build/bin"
TURBO_BIN = Path.home() / "Pictures/llamacpp-turboquant-native"
MODEL_DIR = Path.home()
DEFAULT_MODEL = "Qwen3.6-27B-MTP-pi-tune-Q4_K_M.gguf"
SERVICE_NAME = "llama-server"
QUANT_TYPES = ['Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L', 'Q4_K_M', 'Q4_K_S', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0', 'F16', 'F32']
```
<!-- /AUTO:config -->

## Auto-updating README

`scripts/gen_readme.py` regenerates the `<!-- AUTO:... -->` sections above from the source. A pre-commit hook runs it automatically whenever `llm-shell.py` is staged:

```bash
# install the hook (already done if you cloned this repo)
cp .git/hooks/pre-commit .git/hooks/pre-commit
```

Run manually at any time:

```bash
python3 scripts/gen_readme.py
```

## License

MIT
