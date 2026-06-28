#!/usr/bin/env python3
"""llm-shell: TUI control panel for llama.cpp server, models, and conversion."""
# run with: uv run --with textual llm-shell.py

from __future__ import annotations

import glob
import json
import math
import os
import shutil
import struct
import subprocess
import webbrowser
from pathlib import Path
from typing import ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

# ── Config ──────────────────────────────────────────────────────────────────────

BUILD_BIN = Path.home() / "Documents/ai-assistant/ai-quant-platform/llama.cpp/build/bin"
TURBO_BIN = Path.home() / "Pictures/llamacpp-turboquant-native"
MODEL_DIR  = Path.home()
DEFAULT_MODEL = "Qwen3.6-27B-MTP-pi-tune-Q4_K_M.gguf"
SERVICE_NAME  = "llama-server"

QUANT_TYPES = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_M", "Q4_K_S",
               "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16", "F32"]


def _server_bin() -> Path:
    """Prefer newly-built binary; fall back to turboquant pre-build."""
    built = BUILD_BIN / "llama-server"
    if built.exists():
        return built
    return TURBO_BIN / "llama-server-cuda"


def _quantize_bin() -> Path:
    built = BUILD_BIN / "llama-quantize"
    if built.exists():
        return built
    return TURBO_BIN / "bin" / "llama-quantize"


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args, SERVICE_NAME],
        capture_output=True, text=True,
    )


def _service_active() -> bool:
    r = _systemctl("is-active")
    return r.stdout.strip() == "active"


def _service_enabled() -> bool:
    r = _systemctl("is-enabled")
    return r.stdout.strip() == "enabled"


def _gguf_models() -> list[Path]:
    return sorted(MODEL_DIR.glob("*.gguf"))


# ── Confirm dialog ──────────────────────────────────────────────────────────────

class ConfirmDialog(ModalScreen[bool]):
    BINDINGS: ClassVar = [Binding("escape", "dismiss(False)")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(self._message, id="dialog-msg")
            with Horizontal(id="dialog-btns"):
                yield Button("Confirm", variant="error", id="yes")
                yield Button("Cancel", variant="default", id="no")

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no(self) -> None:
        self.dismiss(False)


# ── Server tab ──────────────────────────────────────────────────────────────────

class ServerTab(Vertical):
    active: reactive[bool] = reactive(False)
    enabled: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        with Horizontal(id="server-status-row"):
            yield Label("Status:", id="status-label")
            yield Label("...", id="status-value")
            yield Label("Auto-start:", id="enabled-label")
            yield Label("...", id="enabled-value")
        with Horizontal(id="server-btns"):
            yield Button("Start",   variant="success", id="btn-start")
            yield Button("Stop",    variant="error",   id="btn-stop")
            yield Button("Restart", variant="warning",  id="btn-restart")
            yield Button("Enable auto-start",  variant="primary", id="btn-enable")
            yield Button("Disable auto-start", variant="default", id="btn-disable")
        yield Label("Live logs (last 80 lines):", classes="section-title")
        yield RichLog(id="server-log", highlight=True, markup=True, max_lines=200)
        with Horizontal(id="server-config-row"):
            yield Label("Port:", classes="cfg-label")
            yield Input("8080", id="cfg-port", classes="cfg-input")
            yield Label("Context:", classes="cfg-label")
            yield Input("16384", id="cfg-ctx", classes="cfg-input")
            yield Label("GPU layers:", classes="cfg-label")
            yield Input("55", id="cfg-ngl", classes="cfg-input")
            yield Label("CPU layers:", classes="cfg-label")
            yield Input("10", id="cfg-cpu", classes="cfg-input")

    def on_mount(self) -> None:
        self._refresh_worker()
        self.set_interval(5, self._refresh_worker)

    def refresh_status(self) -> None:
        self._refresh_worker()

    @work(thread=True)
    def _refresh_worker(self) -> None:
        active  = _service_active()
        enabled = _service_enabled()
        self.app.call_from_thread(self._set_status, active, enabled)

    @on(Button.Pressed, "#btn-start")
    def _btn_start(self) -> None: self._svc_action("start")

    @on(Button.Pressed, "#btn-stop")
    def _btn_stop(self) -> None: self._svc_action("stop")

    @on(Button.Pressed, "#btn-restart")
    def _btn_restart(self) -> None: self._svc_action("restart")

    @on(Button.Pressed, "#btn-enable")
    def _btn_enable(self) -> None: self._svc_action("enable")

    @on(Button.Pressed, "#btn-disable")
    def _btn_disable(self) -> None: self._svc_action("disable")

    @work(thread=True)
    def _svc_action(self, action: str) -> None:
        log: RichLog = self.query_one("#server-log", RichLog)
        log.write(f"[dim]$ systemctl --user {action} {SERVICE_NAME}[/]")
        r = _systemctl(action)
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            log.write(f"[red]Error (exit {r.returncode}):[/] {out or '(no output)'}")
        elif out:
            log.write(out)
        active  = _service_active()
        enabled = _service_enabled()
        self.app.call_from_thread(self._set_status, active, enabled)
        if action in ("start", "restart"):
            import time; time.sleep(1)
            r2 = subprocess.run(
                ["journalctl", "--user", "-u", SERVICE_NAME, "-n", "40", "--no-pager"],
                capture_output=True, text=True,
            )
            for line in r2.stdout.splitlines():
                log.write(line)

    def _set_status(self, active: bool, enabled: bool) -> None:
        self.active  = active
        self.enabled = enabled
        self.query_one("#status-value",  Label).update(
            "[green]active[/]" if active else "[red]inactive[/]"
        )
        self.query_one("#enabled-value", Label).update(
            "[green]enabled[/]" if enabled else "[dim]disabled[/]"
        )


# ── Models tab ──────────────────────────────────────────────────────────────────

class ModelsTab(Vertical):
    def compose(self) -> ComposeResult:
        yield Label("GGUF models in ~/", classes="section-title")
        yield DataTable(id="models-table")
        with Horizontal(id="model-btns"):
            yield Button("Refresh", variant="default", id="btn-refresh-models")
            yield Button("Launch server with selected", variant="success", id="btn-launch-model")
            yield Button("Delete selected", variant="error", id="btn-delete-model")
        yield Log(id="model-log", max_lines=30)

    def on_mount(self) -> None:
        tbl: DataTable = self.query_one("#models-table")
        tbl.add_columns("Model", "Size", "Path")
        self._populate()

    def _populate(self) -> None:
        tbl: DataTable = self.query_one("#models-table")
        tbl.clear()
        for p in _gguf_models():
            size_mb = p.stat().st_size / 1_048_576
            tbl.add_row(p.name, f"{size_mb:,.0f} MB", str(p))

    @on(Button.Pressed, "#btn-refresh-models")
    def _refresh(self) -> None:
        self._populate()

    @on(Button.Pressed, "#btn-launch-model")
    @work(thread=True)
    def _launch(self) -> None:
        tbl: DataTable = self.query_one("#models-table")
        if tbl.cursor_row < 0:
            return
        row = tbl.get_row_at(tbl.cursor_row)
        model_path = str(row[2])
        log: Log = self.query_one("#model-log")
        log.write_line(f"Launching {row[0]} via systemd …")
        _systemctl("stop")
        # Rewrite service ExecStart with new model path
        svc = Path.home() / ".config/systemd/user/llama-server.service"
        text = svc.read_text()
        import re
        text = re.sub(r"-m [^\s]+", f"-m {model_path}", text)
        svc.write_text(text)
        subprocess.run(["systemctl", "--user", "daemon-reload"])
        _systemctl("start")
        log.write_line("Service restarted with new model.")

    @on(Button.Pressed, "#btn-delete-model")
    def _delete(self) -> None:
        tbl: DataTable = self.query_one("#models-table")
        if tbl.cursor_row < 0:
            return
        row = tbl.get_row_at(tbl.cursor_row)

        def _do_delete(confirmed: bool) -> None:
            if confirmed:
                Path(str(row[2])).unlink(missing_ok=True)
                self._populate()

        self.app.push_screen(ConfirmDialog(f"Delete {row[0]}?"), _do_delete)


# ── Convert/Quantize tab ─────────────────────────────────────────────────────────

class ConvertTab(Vertical):
    def compose(self) -> ComposeResult:
        yield Label("Quantize a GGUF model", classes="section-title")
        with Horizontal(classes="form-row"):
            yield Label("Input GGUF:", classes="form-label")
            yield Select(
                [(p.name, str(p)) for p in _gguf_models()],
                id="sel-input", prompt="Select model …",
            )
        with Horizontal(classes="form-row"):
            yield Label("Output name:", classes="form-label")
            yield Input("output.gguf", id="txt-output", classes="form-input")
        with Horizontal(classes="form-row"):
            yield Label("Quant type:", classes="form-label")
            yield Select(
                [(q, q) for q in QUANT_TYPES],
                value="Q4_K_M", id="sel-quant",
            )
        yield Button("Quantize", variant="success", id="btn-quantize")
        yield Label("─" * 60)
        yield Label("Convert HuggingFace model → GGUF", classes="section-title")
        with Horizontal(classes="form-row"):
            yield Label("HF model dir:", classes="form-label")
            yield Input(str(Path.home()), id="txt-hf-dir", classes="form-input")
        with Horizontal(classes="form-row"):
            yield Label("Output type:", classes="form-label")
            yield Select([("F16","f16"),("BF16","bf16"),("F32","f32")],
                         value="f16", id="sel-hf-type")
        yield Button("Convert to GGUF", variant="primary", id="btn-convert")
        yield Label("Output:", classes="section-title")
        yield Log(id="convert-log", max_lines=100)

    @on(Button.Pressed, "#btn-quantize")
    @work(thread=True)
    def _quantize(self) -> None:
        log: Log = self.query_one("#convert-log")
        inp = self.query_one("#sel-input", Select).value
        out = self.query_one("#txt-output", Input).value.strip()
        qtype = self.query_one("#sel-quant", Select).value

        if inp is Select.BLANK or not out:
            log.write_line("Select an input model and output name first.")
            return

        out_path = MODEL_DIR / out
        qbin = _quantize_bin()
        if not qbin.exists():
            log.write_line(f"quantize binary not found: {qbin}")
            return

        log.write_line(f"Quantizing {inp} → {out_path} ({qtype}) …")
        cmd = [str(qbin), inp, str(out_path), qtype]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout
        for line in proc.stdout:
            log.write_line(line.rstrip())
        proc.wait()
        log.write_line(f"Done (exit {proc.returncode}).")

    @on(Button.Pressed, "#btn-convert")
    @work(thread=True)
    def _convert(self) -> None:
        log: Log = self.query_one("#convert-log")
        hf_dir = self.query_one("#txt-hf-dir", Input).value.strip()
        outtype = self.query_one("#sel-hf-type", Select).value

        script = BUILD_BIN.parent.parent / "convert_hf_to_gguf.py"
        if not script.exists():
            script = TURBO_BIN / "convert_hf_to_gguf.py"
        if not script.exists():
            log.write_line("convert_hf_to_gguf.py not found in build dir.")
            return

        out_path = Path(hf_dir) / f"model-{outtype}.gguf"
        log.write_line(f"Converting {hf_dir} → {out_path} …")
        cmd = ["python3", str(script), hf_dir, "--outtype", outtype, "--outfile", str(out_path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout
        for line in proc.stdout:
            log.write_line(line.rstrip())
        proc.wait()
        log.write_line(f"Done (exit {proc.returncode}).")


# ── Run tab (launch without systemd) ────────────────────────────────────────────

class RunTab(Vertical):
    _proc: subprocess.Popen | None = None

    def compose(self) -> ComposeResult:
        yield Label("Launch server directly (foreground, MTP speculative decoding)", classes="section-title")
        with Horizontal(classes="form-row"):
            yield Label("Model:", classes="form-label")
            yield Select(
                [(p.name, str(p)) for p in _gguf_models()],
                value=str(MODEL_DIR / DEFAULT_MODEL) if (MODEL_DIR / DEFAULT_MODEL).exists() else Select.BLANK,
                id="run-model",
            )
        with Horizontal(classes="form-row"):
            yield Label("Context:", classes="form-label")
            yield Input("16384", id="run-ctx", classes="form-input-sm")
            yield Label("GPU layers:", classes="form-label")
            yield Input("55", id="run-ngl", classes="form-input-sm")
            yield Label("CPU layers:", classes="form-label")
            yield Input("10", id="run-cpu", classes="form-input-sm")
            yield Label("Port:", classes="form-label")
            yield Input("8080", id="run-port", classes="form-input-sm")
        with Horizontal(classes="form-row"):
            yield Label("MTP drafts:", classes="form-label")
            yield Input("3", id="run-mtp", classes="form-input-sm")
            yield Label("KV cache type:", classes="form-label")
            yield Select([("q8_0","q8_0"),("q4_0","q4_0"),("f16","f16")],
                         value="q8_0", id="run-kv")
        with Horizontal(classes="form-row"):
            yield Label("Extra flags:", classes="form-label")
            yield Input("", id="run-extra", classes="form-input")
        with Horizontal(id="run-btns"):
            yield Button("Start",       variant="success", id="btn-run-start")
            yield Button("Stop",        variant="error",   id="btn-run-stop")
            yield Button("Auto Config", variant="warning", id="btn-run-auto")
        yield Label("Server output:", classes="section-title")
        yield Log(id="run-log", max_lines=500)

    @on(Button.Pressed, "#btn-run-start")
    def _on_run_start(self) -> None:
        self._launch_server()

    @on(Button.Pressed, "#btn-run-stop")
    def _on_run_stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self.query_one("#run-log", Log).write_line("[Sent SIGTERM]")

    @on(Button.Pressed, "#btn-run-auto")
    def _on_auto_pressed(self) -> None:
        self._do_auto_config()

    @work(thread=True)
    def _do_auto_config(self) -> None:
        log: Log = self.query_one("#run-log")
        model = self.query_one("#run-model", Select).value
        if model is Select.BLANK:
            log.write_line("Select a model first.")
            return
        log.write_line("Reading model metadata and detecting hardware …")
        hw  = _detect_hw()
        cfg = _auto_config(model, hw)
        self.app.call_from_thread(self._apply_cfg, cfg)
        log.write_line(
            f"  {Path(model).name}  —  {cfg['model_gb']:.1f} GB on disk, "
            f"{cfg['block_count']} transformer layers, max ctx {cfg['max_ctx']:,}"
        )
        log.write_line(
            f"  GPU: {hw['gpu']}  ({hw['vram_mb'] / 1024:.1f} GB VRAM, "
            f"{hw['ram_mb'] / 1024:.0f} GB RAM)"
        )
        log.write_line(
            f"  → GPU layers: {cfg['ngl']}  CPU layers: {cfg['cpu']}  "
            f"Context: {cfg['ctx']}"
        )

    def _apply_cfg(self, cfg: dict) -> None:
        self.query_one("#run-ngl", Input).value = cfg["ngl"]
        self.query_one("#run-cpu", Input).value = cfg["cpu"]
        self.query_one("#run-ctx", Input).value = cfg["ctx"]

    @work(thread=True)
    def _launch_server(self) -> None:
        log: Log = self.query_one("#run-log")
        log.write_line("── launching server ──")

        model = self.query_one("#run-model", Select).value
        if model is Select.BLANK or not isinstance(model, str):
            log.write_line("[ERROR] No model selected.")
            return
        if not Path(model).exists():
            log.write_line(f"[ERROR] Model not found: {model}")
            return

        bin_path = _server_bin()
        if not bin_path.exists():
            log.write_line(f"[ERROR] Server binary not found: {bin_path}")
            log.write_line("  Build llama.cpp or check BUILD_BIN in the script.")
            return

        ctx     = self.query_one("#run-ctx",   Input).value or "16384"
        ngl     = self.query_one("#run-ngl",   Input).value or "55"
        cpu_lyr = self.query_one("#run-cpu",   Input).value.strip() or "10"
        port    = self.query_one("#run-port",  Input).value or "8080"
        mtp     = self.query_one("#run-mtp",   Input).value or "3"
        kv      = self.query_one("#run-kv",    Select).value or "q8_0"
        extra   = self.query_one("#run-extra", Input).value.split()

        try:
            effective_ngl = str(max(0, int(ngl) - int(cpu_lyr)))
        except ValueError:
            effective_ngl = ngl

        env = os.environ.copy()
        lib_dir = str(BUILD_BIN)
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing_ld}" if existing_ld else lib_dir

        if int(cpu_lyr) > 0:
            log.write_line(f"GPU layers: {effective_ngl}  CPU layers: {cpu_lyr}")
        cmd = [
            str(bin_path),
            "-m", model,
            "--jinja", "-ngl", effective_ngl, "--flash-attn", "on", "-c", ctx,
            "--spec-type", "draft-mtp", "--spec-draft-n-max", mtp,
            "--cache-type-k", kv, "--cache-type-v", kv,
            "--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0",
            "--presence-penalty", "1.5",
            "--host", "127.0.0.1", "--port", port,
            *extra,
        ]
        log.write_line("$ " + " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env,
            )
        except Exception as exc:
            log.write_line(f"[ERROR] Failed to start process: {exc}")
            return
        assert self._proc.stdout
        for line in self._proc.stdout:
            log.write_line(line.rstrip())
        self._proc.wait()
        log.write_line(f"── server exited (code {self._proc.returncode}) ──")
        self._proc = None


# ── Hardware detection ──────────────────────────────────────────────────────────

def _detect_hw() -> dict:
    vram_mb, ram_mb = 0, 0
    gpu_name = "Unknown GPU"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            gpu_name = parts[0].strip()
            vram_mb  = int(parts[1].strip().split()[0])
    except FileNotFoundError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    ram_mb = int(line.split()[1]) // 1024
                    break
    except OSError:
        pass
    return {"gpu": gpu_name, "vram_mb": vram_mb, "ram_mb": ram_mb}


def _read_gguf_metadata(path: Path) -> dict:
    """Parse model architecture metadata from a GGUF file header (no external deps)."""
    result: dict = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return result
            f.read(4)   # version
            f.read(8)   # tensor_count
            kv_count = struct.unpack("<Q", f.read(8))[0]

            SCALAR = {0:("<B",1),1:("<b",1),2:("<H",2),3:("<h",2),
                      4:("<I",4),5:("<i",4),6:("<f",4),7:("<B",1),
                      10:("<Q",8),11:("<q",8),12:("<d",8)}

            def read_str() -> str:
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", errors="replace")

            def read_val(vtype: int):
                if vtype in SCALAR:
                    fmt, sz = SCALAR[vtype]
                    return struct.unpack(fmt, f.read(sz))[0]
                if vtype == 8:
                    return read_str()
                if vtype == 9:  # array — must consume all elements to keep position
                    et = struct.unpack("<I", f.read(4))[0]
                    n  = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(n):
                        read_val(et)
                    return None
                return None

            wanted = {"block_count", "context_length", "embedding_length",
                      "attention.head_count", "attention.head_count_kv"}

            for _ in range(kv_count):
                key   = read_str()
                vtype = struct.unpack("<I", f.read(4))[0]
                val   = read_val(vtype)
                # Strip arch prefix (llama., qwen2., mistral., …) → portable suffix
                dot    = key.find(".")
                suffix = key[dot+1:] if dot != -1 else key
                if suffix in wanted and val is not None:
                    result[suffix] = val
    except Exception:
        pass
    return result


def _auto_config(model_path: str, hw: dict) -> dict:
    """Return recommended {ngl, cpu, ctx} for the model + detected hardware."""
    path     = Path(model_path)
    meta     = _read_gguf_metadata(path)

    vram_gb  = hw["vram_mb"] / 1024
    model_gb = path.stat().st_size / 1024**3

    block_count = int(meta.get("block_count",             32))
    max_ctx     = int(meta.get("context_length",       32768))
    kv_heads    = int(meta.get("attention.head_count_kv",  8))
    head_count  = int(meta.get("attention.head_count",    32))
    emb_len     = int(meta.get("embedding_length",      4096))
    head_dim    = emb_len // max(head_count, 1)

    avail_vram = vram_gb * 0.88
    layer_gb   = model_gb / (block_count + 2)  # +2 for embed/output layers

    # KV cache with q8_0: ≈1 byte per element (GPU layers only for on-GPU KV)
    kv_per_tok_gb = 2 * block_count * kv_heads * head_dim / 1024**3

    # Reserve VRAM for at least 8K context before fitting GPU layers
    min_kv_gb  = max(1.0, kv_per_tok_gb * 8192)
    budget     = max(0.0, avail_vram - min_kv_gb)
    gpu_layers = min(block_count, int(budget / layer_gb))
    cpu_layers = max(0, block_count - gpu_layers)

    remaining = avail_vram - gpu_layers * layer_gb
    if kv_per_tok_gb > 0 and remaining > 0:
        raw_ctx = int(remaining * 0.80 / kv_per_tok_gb)
    else:
        raw_ctx = 4096
    ctx = max(4096, (min(max_ctx, max(4096, raw_ctx)) // 4096) * 4096)

    return {
        "ngl":         str(gpu_layers),
        "cpu":         str(cpu_layers),
        "ctx":         str(ctx),
        "block_count": block_count,
        "model_gb":    model_gb,
        "max_ctx":     max_ctx,
    }


def _quant_gb(params_b: float, bits: float) -> float:
    """Estimated model size in GB for given param count and bits-per-weight."""
    return params_b * 1e9 * bits / 8 / 1024**3


# bits per weight for each quant type
_QUANT_BITS = {
    "Q2_K": 2.6, "Q3_K_S": 3.0, "Q3_K_M": 3.35, "Q3_K_L": 3.6,
    "Q4_K_S": 4.0, "Q4_K_M": 4.5, "Q5_K_S": 5.0, "Q5_K_M": 5.5,
    "Q6_K": 6.6, "Q8_0": 8.5, "F16": 16.0, "BF16": 16.0, "F32": 32.0,
}


# Curated model catalogue  {name, params_b, hf_repo, description, tags}
_MODEL_CATALOGUE = [
    # ── Coding ─────────────────────────────────────────────────────────────────
    {"name": "Qwen2.5-Coder-7B",   "params": 7.6,  "hf": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",       "desc": "Best small coding model",          "tags": "code"},
    {"name": "Qwen2.5-Coder-14B",  "params": 14.8, "hf": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",      "desc": "Strong code + reasoning",           "tags": "code"},
    {"name": "Qwen2.5-Coder-32B",  "params": 32.8, "hf": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",      "desc": "Top open coding model",             "tags": "code"},
    {"name": "DeepSeek-Coder-V2-Lite","params":16,  "hf": "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF","desc":"MoE coding, 16B active params",    "tags": "code"},
    {"name": "Codestral-22B",       "params": 22.2, "hf": "bartowski/Codestral-22B-v0.1-GGUF",          "desc": "Mistral's flagship code model",     "tags": "code"},
    # ── General / Chat ─────────────────────────────────────────────────────────
    {"name": "Llama-3.1-8B",        "params": 8.0,  "hf": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",  "desc": "Meta's solid small general model",  "tags": "chat"},
    {"name": "Llama-3.3-70B",       "params": 70.6, "hf": "bartowski/Llama-3.3-70B-Instruct-GGUF",      "desc": "Meta's best open model (CPU offload)","tags":"chat"},
    {"name": "Mistral-7B-v0.3",     "params": 7.2,  "hf": "TheBloke/Mistral-7B-Instruct-v0.3-GGUF",    "desc": "Fast, efficient, well-tested",       "tags": "chat"},
    {"name": "Mistral-Small-24B",   "params": 24.0, "hf": "bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF","desc":"Mistral's efficient mid-size model","tags":"chat"},
    {"name": "Gemma-3-12B",         "params": 12.0, "hf": "bartowski/gemma-3-12b-it-GGUF",               "desc": "Google's balanced multimodal model","tags": "chat,vision"},
    {"name": "Gemma-3-27B",         "params": 27.0, "hf": "bartowski/gemma-3-27b-it-GGUF",               "desc": "Google's strongest open model",     "tags": "chat,vision"},
    {"name": "Phi-4-14B",           "params": 14.0, "hf": "bartowski/phi-4-GGUF",                        "desc": "Microsoft's reasoning-focused model","tags": "chat,reason"},
    {"name": "Phi-4-mini-4B",       "params": 3.8,  "hf": "bartowski/Phi-4-mini-instruct-GGUF",          "desc": "Tiny but punchy reasoning model",   "tags": "chat,reason"},
    {"name": "Qwen3-8B",            "params": 8.2,  "hf": "Qwen/Qwen3-8B-GGUF",                         "desc": "Latest Qwen3 with thinking mode",   "tags": "chat,reason"},
    {"name": "Qwen3-14B",           "params": 14.8, "hf": "Qwen/Qwen3-14B-GGUF",                        "desc": "Qwen3 mid-size, great all-rounder", "tags": "chat,reason"},
    {"name": "Qwen3-32B",           "params": 32.8, "hf": "Qwen/Qwen3-32B-GGUF",                        "desc": "Qwen3 flagship (CPU offload needed)","tags": "chat,reason"},
    # ── Reasoning / Math ───────────────────────────────────────────────────────
    {"name": "DeepSeek-R1-Distill-8B",  "params": 8.0,  "hf": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF", "desc":"R1 reasoning distilled to 8B",   "tags": "reason,math"},
    {"name": "DeepSeek-R1-Distill-14B", "params": 14.0, "hf": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF", "desc":"R1 reasoning distilled to 14B",  "tags": "reason,math"},
    {"name": "QwQ-32B",             "params": 32.8, "hf": "Qwen/QwQ-32B-GGUF",                          "desc": "Qwen reasoning model (CPU offload)","tags": "reason,math"},
    # ── Small / Fast ───────────────────────────────────────────────────────────
    {"name": "SmolLM2-1.7B",        "params": 1.7,  "hf": "bartowski/SmolLM2-1.7B-Instruct-GGUF",       "desc": "Ultra-fast tiny model",             "tags": "fast"},
    {"name": "Llama-3.2-3B",        "params": 3.2,  "hf": "bartowski/Llama-3.2-3B-Instruct-GGUF",       "desc": "Small Meta model, great on CPU",    "tags": "fast,chat"},
    {"name": "Gemma-3-4B",          "params": 4.0,  "hf": "bartowski/gemma-3-4b-it-GGUF",                "desc": "Tiny but surprisingly capable",     "tags": "fast,chat"},
    # ── Vision / Multimodal ────────────────────────────────────────────────────
    {"name": "LLaVA-v1.6-Mistral-7B","params": 7.2, "hf": "cjpais/llava-v1.6-mistral-7b-gguf",          "desc": "Image + text understanding",        "tags": "vision"},
    {"name": "Qwen2.5-VL-7B",       "params": 8.3,  "hf": "bartowski/Qwen2.5-VL-7B-Instruct-GGUF",      "desc": "Strong vision-language model",      "tags": "vision"},
]


def _fit_status(params_b: float, vram_mb: int, ram_mb: int, quant: str) -> tuple[str, str]:
    """Return (emoji_label, detail) for how well a model fits the hardware."""
    bits = _QUANT_BITS.get(quant, 4.5)
    size_gb = _quant_gb(params_b, bits)
    vram_gb = vram_mb / 1024
    ram_gb  = ram_mb  / 1024
    # KV cache headroom estimate
    avail_vram = vram_gb * 0.88

    if size_gb <= avail_vram:
        return "✅ GPU", f"{size_gb:.1f} GB — fully on GPU"
    elif size_gb <= avail_vram + ram_gb * 0.5:
        gpu_layers = max(1, int(avail_vram / size_gb * 99))
        return "🔀 Hybrid", f"{size_gb:.1f} GB — ~{gpu_layers} layers on GPU"
    else:
        return "❌ Too large", f"{size_gb:.1f} GB — exceeds VRAM+RAM"


# ── Model Finder tab ────────────────────────────────────────────────────────────

class ModelFinderTab(Vertical):
    def compose(self) -> ComposeResult:
        self._hw = _detect_hw()
        vram_gb = self._hw["vram_mb"] / 1024
        ram_gb  = self._hw["ram_mb"]  / 1024
        yield Static(
            f"[bold]{self._hw['gpu']}[/]  ·  "
            f"[green]{vram_gb:.1f} GB VRAM[/]  ·  "
            f"[cyan]{ram_gb:.0f} GB RAM[/]",
            id="hw-banner",
        )
        with Horizontal(id="finder-controls"):
            yield Label("Filter:", classes="form-label")
            yield Select(
                [("All", "all"), ("✅ Fits on GPU", "gpu"), ("🔀 Hybrid", "hybrid"),
                 ("Code", "code"), ("Chat", "chat"), ("Reasoning", "reason"),
                 ("Vision", "vision"), ("Fast/Small", "fast")],
                value="gpu", id="sel-filter",
            )
            yield Label("Quant:", classes="form-label")
            yield Select([(q, q) for q in QUANT_TYPES], value="Q4_K_M", id="sel-finder-quant")
            yield Button("Refresh", variant="default", id="btn-finder-refresh")
        yield DataTable(id="finder-table")
        with Horizontal(id="finder-btns"):
            yield Button("Open on HuggingFace ↗", variant="primary",   id="btn-hf-open")
            yield Button("Download (hf-cli)",      variant="success",   id="btn-hf-download")
            yield Button("Copy ollama pull cmd",   variant="default",   id="btn-ollama-copy")
        yield Log(id="finder-log", max_lines=30)

    def on_mount(self) -> None:
        tbl: DataTable = self.query_one("#finder-table")
        tbl.add_columns("Fit", "Model", "Params", f"Size (Q4_K_M)", "Tags", "Description")
        self._populate()

    def _populate(self) -> None:
        tbl: DataTable = self.query_one("#finder-table")
        tbl.clear()
        filt  = self.query_one("#sel-filter",       Select).value
        quant = self.query_one("#sel-finder-quant", Select).value

        # Update size column header
        tbl.columns  # access to update label if needed

        for m in _MODEL_CATALOGUE:
            fit_label, fit_detail = _fit_status(
                m["params"], self._hw["vram_mb"], self._hw["ram_mb"], quant
            )
            # Apply filter
            if filt == "gpu"    and "GPU"    not in fit_label: continue
            if filt == "hybrid" and "Hybrid" not in fit_label: continue
            if filt in ("code","chat","reason","vision","fast") and filt not in m["tags"]: continue

            bits = _QUANT_BITS.get(quant, 4.5)
            size_gb = _quant_gb(m["params"], bits)
            tbl.add_row(
                fit_label,
                m["name"],
                f"{m['params']:.1f}B",
                f"{size_gb:.1f} GB",
                m["tags"],
                m["desc"],
            )

    @on(Select.Changed, "#sel-filter")
    @on(Select.Changed, "#sel-finder-quant")
    def _on_filter(self) -> None:
        self._populate()

    @on(Button.Pressed, "#btn-finder-refresh")
    def _refresh(self) -> None:
        self._hw = _detect_hw()
        vram_gb = self._hw["vram_mb"] / 1024
        ram_gb  = self._hw["ram_mb"]  / 1024
        self.query_one("#hw-banner", Static).update(
            f"[bold]{self._hw['gpu']}[/]  ·  "
            f"[green]{vram_gb:.1f} GB VRAM[/]  ·  "
            f"[cyan]{ram_gb:.0f} GB RAM[/]"
        )
        self._populate()

    def _selected_model(self) -> dict | None:
        tbl: DataTable = self.query_one("#finder-table")
        if tbl.cursor_row < 0:
            return None
        row = tbl.get_row_at(tbl.cursor_row)
        name = str(row[1])
        return next((m for m in _MODEL_CATALOGUE if m["name"] == name), None)

    @on(Button.Pressed, "#btn-hf-open")
    def _open_hf(self) -> None:
        m = self._selected_model()
        if m:
            webbrowser.open(f"https://huggingface.co/{m['hf']}")

    @on(Button.Pressed, "#btn-hf-download")
    @work(thread=True)
    def _download(self) -> None:
        m = self._selected_model()
        log: Log = self.query_one("#finder-log")
        if not m:
            log.write_line("Select a model first.")
            return
        quant = self.query_one("#sel-finder-quant", Select).value
        # Try to find a matching GGUF file pattern
        pattern = f"*{quant}*.gguf"
        log.write_line(f"Downloading {m['name']} ({quant}) from {m['hf']} …")
        log.write_line("This may take a while depending on model size.")
        cmd = [
            "uv", "run", "--with", "huggingface_hub[cli]",
            "huggingface-cli", "download", m["hf"],
            "--include", pattern,
            "--local-dir", str(MODEL_DIR),
        ]
        log.write_line("$ " + " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout
        for line in proc.stdout:
            log.write_line(line.rstrip())
        proc.wait()
        log.write_line(f"Done (exit {proc.returncode}).")

    @on(Button.Pressed, "#btn-ollama-copy")
    def _ollama_copy(self) -> None:
        m = self._selected_model()
        log: Log = self.query_one("#finder-log")
        if not m:
            log.write_line("Select a model first.")
            return
        # Build an approximate ollama model name
        slug = m["name"].lower().replace("-instruct","").replace(".","-")
        cmd = f"ollama pull {slug}"
        try:
            subprocess.run(["xclip", "-selection", "clipboard"], input=cmd, text=True, check=True)
            log.write_line(f"Copied: {cmd}")
        except Exception:
            log.write_line(f"Run: {cmd}")


# ── Main App ─────────────────────────────────────────────────────────────────────

CSS = """
Screen { background: $surface; }
Header { background: $primary; color: $text; }

#server-status-row, #server-btns, #server-config-row,
#model-btns, #run-btns { height: auto; margin: 1 0; }

.section-title { margin: 1 0 0 0; color: $accent; text-style: bold; }
.form-row { height: auto; margin: 0 0 1 0; align: left middle; }
.form-label { width: 16; }
.form-input { width: 40; }
.form-input-sm { width: 10; }
.cfg-label { width: 12; }
.cfg-input { width: 10; }

#server-log, #model-log, #convert-log, #run-log {
    height: 1fr; min-height: 10; border: solid $primary-darken-2;
    background: $surface-darken-2;
}

#models-table { height: 1fr; min-height: 8; }

#dialog {
    width: 60; height: auto; padding: 2 4;
    background: $surface; border: solid $primary;
    align: center middle;
}
#dialog-msg { margin-bottom: 2; text-align: center; }
#dialog-btns { align: center middle; height: auto; }
#dialog-btns Button { margin: 0 2; }

Button { margin: 0 1; }

#hw-banner {
    background: $primary-darken-3; padding: 1 2; margin-bottom: 1;
    text-align: center; color: $text;
}
#finder-controls { height: auto; margin-bottom: 1; align: left middle; }
#finder-table { height: 1fr; min-height: 10; }
#finder-btns { height: auto; margin: 1 0; }
#finder-log { height: 8; border: solid $primary-darken-2; background: $surface-darken-2; }
"""


class LlamaShell(App):
    TITLE = "llm-shell · llama.cpp control panel"
    CSS = CSS
    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-server"):
            with TabPane("Server",             id="tab-server"):
                yield ServerTab()
            with TabPane("Model Finder",       id="tab-finder"):
                yield ModelFinderTab()
            with TabPane("Models",             id="tab-models"):
                yield ModelsTab()
            with TabPane("Run (direct)",       id="tab-run"):
                yield RunTab()
            with TabPane("Convert / Quantize", id="tab-convert"):
                yield ConvertTab()
        yield Footer()

    def action_refresh(self) -> None:
        self.query_one(ServerTab).refresh_status()


if __name__ == "__main__":
    LlamaShell().run()
