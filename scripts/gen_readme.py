#!/usr/bin/env python3
"""Regenerate AUTO-tagged sections of README.md from llm-shell.py source.

Sections are delimited by:
  <!-- AUTO:<name> -->
  ...generated content...
  <!-- /AUTO:<name> -->

Run directly or via the pre-commit hook.
"""

import re
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
SRC    = ROOT / "llm-shell.py"
README = ROOT / "README.md"

# Prose descriptions for each tab (keyed by the TabPane label string).
# Add/edit here when a tab's purpose changes; names are auto-extracted from source.
TAB_DESCRIPTIONS: dict[str, str] = {
    "Server":             "Start/stop/restart the `llama-server` systemd user service; tail live logs; edit port, context size, and GPU/CPU layer counts before restarting",
    "Model Finder":       "Browse GGUF models on HuggingFace filtered by VRAM budget, open on HF, or copy an `ollama pull` command",
    "Models":             "List every `.gguf` in `~/`, show size and estimated VRAM fit, launch directly or delete",
    "Run (direct)":       "Launch `llama-server` in the foreground without systemd; **Auto Config** reads model metadata and your GPU's free VRAM to suggest optimal GPU/CPU layer split and context length",
    "Convert / Quantize": "Quantize any GGUF to a smaller quant type; convert a HuggingFace checkpoint to GGUF",
}


# ── Parsers ──────────────────────────────────────────────────────────────────

def extract_tabs(src: str) -> list[tuple[str, str]]:
    """Return [(label, tab_id), ...] in declaration order."""
    return re.findall(r'with TabPane\("([^"]+)"\s*,\s*id="([^"]+)"', src)


def extract_tab_sections(src: str, tab_class: str) -> list[str]:
    """Return section-title Label strings defined inside a tab class."""
    # Find the class block (up to next top-level class or EOF)
    m = re.search(rf"^class {re.escape(tab_class)}\b", src, re.MULTILINE)
    if not m:
        return []
    start = m.start()
    nxt = re.search(r"^class \w", src[start + 1:], re.MULTILINE)
    block = src[start: start + nxt.start() + 1] if nxt else src[start:]
    return re.findall(r'yield Label\("([^"]+)"\s*,\s*classes="section-title"', block)


def extract_buttons(src: str, tab_class: str) -> list[str]:
    """Return button labels defined inside a tab class."""
    m = re.search(rf"^class {re.escape(tab_class)}\b", src, re.MULTILINE)
    if not m:
        return []
    start = m.start()
    nxt = re.search(r"^class \w", src[start + 1:], re.MULTILINE)
    block = src[start: start + nxt.start() + 1] if nxt else src[start:]
    return re.findall(r'yield Button\("([^"]+)"', block)


_CONFIG_NAMES = {"BUILD_BIN", "TURBO_BIN", "MODEL_DIR", "DEFAULT_MODEL", "SERVICE_NAME", "QUANT_TYPES"}

def extract_config_constants(src: str) -> list[tuple[str, str]]:
    """Return [(NAME, raw_value_string), ...] for known config constants."""
    results = []
    for m in re.finditer(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.+)$", src, re.MULTILINE):
        name, val = m.group(1), m.group(2).strip()
        if name not in _CONFIG_NAMES:
            continue
        results.append((name, val))
    return results


def extract_quant_types(src: str) -> list[str]:
    m = re.search(r"^QUANT_TYPES\s*=\s*\[([^\]]+)\]", src, re.MULTILINE | re.DOTALL)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


# ── Tab-class mapping ─────────────────────────────────────────────────────────

# Maps TabPane id → class name (brittle if structure changes, but detectable)
TAB_ID_TO_CLASS: dict[str, str] = {
    "tab-server":  "ServerTab",
    "tab-finder":  "ModelFinderTab",
    "tab-models":  "ModelsTab",
    "tab-run":     "RunTab",
    "tab-convert": "ConvertTab",
}


# ── Section generators ────────────────────────────────────────────────────────

def gen_tabs_table(src: str) -> str:
    tabs = extract_tabs(src)
    lines = ["| Tab | Sections | Buttons | Description |",
             "|-----|----------|---------|-------------|"]
    for label, tab_id in tabs:
        cls      = TAB_ID_TO_CLASS.get(tab_id, "")
        sections = extract_tab_sections(src, cls) if cls else []
        buttons  = extract_buttons(src, cls) if cls else []
        desc     = TAB_DESCRIPTIONS.get(label, "")
        sec_str  = ", ".join(f"`{s}`" for s in sections) if sections else "—"
        btn_str  = ", ".join(f"**{b}**" for b in buttons) if buttons else "—"
        lines.append(f"| **{label}** | {sec_str} | {btn_str} | {desc} |")
    return "\n".join(lines)


def gen_config_block(src: str) -> str:
    consts = extract_config_constants(src)
    quants = extract_quant_types(src)
    lines  = ["```python"]
    for name, val in consts:
        if name == "QUANT_TYPES":
            lines.append(f'QUANT_TYPES = {repr(quants)}')
        else:
            lines.append(f"{name} = {val}")
    lines.append("```")
    return "\n".join(lines)


def gen_quant_list(src: str) -> str:
    quants = extract_quant_types(src)
    return ", ".join(f"`{q}`" for q in quants)


GENERATORS = {
    "tabs":   gen_tabs_table,
    "config": gen_config_block,
    "quants": gen_quant_list,
}


# ── README rewriter ───────────────────────────────────────────────────────────

AUTO_RE = re.compile(
    r"(<!-- AUTO:(\w+) -->\n)(.+?)(<!-- /AUTO:\2 -->)",
    re.DOTALL,
)


def update_readme(readme_path: Path, src: str) -> bool:
    """Replace AUTO sections. Returns True if the file changed."""
    original = readme_path.read_text()

    def replacer(m: re.Match) -> str:
        open_tag, key, _old, close_tag = m.group(1), m.group(2), m.group(3), m.group(4)
        gen = GENERATORS.get(key)
        if gen is None:
            return m.group(0)
        new_content = gen(src) + "\n"
        return f"{open_tag}{new_content}{close_tag}"

    updated = AUTO_RE.sub(replacer, original)
    if updated == original:
        return False
    readme_path.write_text(updated)
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        return 1
    if not README.exists():
        print(f"ERROR: {README} not found", file=sys.stderr)
        return 1

    src = SRC.read_text()
    changed = update_readme(README, src)
    if changed:
        print(f"README.md updated")
    else:
        print(f"README.md unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
