# llm-shell

A terminal UI control panel for [llama.cpp](https://github.com/ggerganov/llama.cpp) — manage your local inference server, browse models, quantize, and convert, all from one keyboard-driven interface.

Built with [Textual](https://textual.textualize.io/).

## Screenshots

```
┌─────────────────────────────────────────────────────────────────┐
│  llm-shell                                             [q] quit │
├──────────┬─────────────┬────────┬─────────────┬────────────────┤
│  Server  │ Model Finder│ Models │ Run (direct)│ Convert/Quant  │
├──────────┴─────────────┴────────┴─────────────┴────────────────┤
│ ● active  enabled   [Start] [Stop] [Restart] [Enable] [Disable] │
│                                                                 │
│ Live logs ...                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Tabs

| Tab | What it does |
|-----|-------------|
| **Server** | Start/stop/restart the `llama-server` systemd user service; tail live logs; tweak port, context, and GPU layer count before restarting |
| **Model Finder** | Browse GGUF models on HuggingFace filtered by VRAM, open on HF, or copy an `ollama pull` command |
| **Models** | List every `.gguf` file in `~/`, show size and estimated VRAM fit, launch or delete |
| **Run (direct)** | Launch `llama-server` in the foreground without systemd; includes **Auto Config** which reads the model's metadata and calculates the optimal GPU/CPU layer split and context size for your GPU |
| **Convert / Quantize** | Quantize any GGUF to a smaller quant type, or convert a HuggingFace checkpoint to GGUF |

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

```python
BUILD_BIN    = Path.home() / "..."   # path to llama.cpp build/bin
TURBO_BIN    = Path.home() / "..."   # fallback binary directory
MODEL_DIR    = Path.home()           # directory scanned for .gguf files
DEFAULT_MODEL = "model.gguf"
SERVICE_NAME  = "llama-server"       # systemd unit name
```

## License

MIT
