"""
Dynamic prompt loader.

All LLM prompt TEXT for the controllers lives under the top-level
`prompts/` folder as plain .txt template files, one prompt per file,
grouped into a subfolder per controller (prompts/mutual_fund/,
prompts/nifty_stock/, prompts/option/, prompts/swing/, prompts/stock/).

Keeping the prompt copy out of the .py files means it can be edited,
reviewed, or removed without touching any code -- and non-engineers can
tweak prompt wording without going near the controller logic.

Usage
-----
    from utils.prompt_loader import load_prompt

    prompt = load_prompt(
        "mutual_fund/market",
        today_str=today_str,
        lookback_note=lookback_note,
        topic_list=topic_list,
        SOURCE_QUALITY_NOTE=SOURCE_QUALITY_NOTE,
        NO_FABRICATION_NOTE=NO_FABRICATION_NOTE,
    )

`name` is the template's path relative to the `prompts/` folder, without
the .txt extension (e.g. "mutual_fund/market" -> prompts/mutual_fund/market.txt).

Templates use plain str.format() placeholders (`{today_str}`), so any
literal `{`/`}` in a template (e.g. JSON schema examples) must be escaped
by doubling it (`{{` / `}}`) exactly as it would be in an f-string.

Templates are cached in-memory after their first read (see
`clear_prompt_cache` / `reload=True` below) so repeated calls in the same
run don't re-hit disk, while still requiring nothing more than editing a
.txt file to change what gets sent to the model on the next run.
"""

import os

# Project root = the parent directory of this file's `utils/` folder.
_PROMPTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
)

_cache = {}


def _template_path(name):
    # Normalize "mutual_fund/market" (or "mutual_fund\\market") to a real
    # filesystem path under prompts/, and guard against path escapes.
    rel = name.replace("\\", "/").strip("/")
    if ".." in rel.split("/"):
        raise ValueError(f"Invalid prompt name (path traversal not allowed): {name!r}")
    return os.path.join(_PROMPTS_ROOT, *rel.split("/")) + ".txt"


def _read_template(name, reload=False):
    if not reload and name in _cache:
        return _cache[name]
    path = _template_path(name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Prompt template '{name}' not found at {path}. "
            f"Expected a .txt file under the prompts/ folder."
        )
    _cache[name] = text
    return text


def load_prompt(name, reload=False, **kwargs):
    """
    Loads prompts/<name>.txt and fills in {placeholder} values from kwargs
    via str.format(). Any literal `{`/`}` in the template (e.g. JSON
    schema braces) must already be doubled (`{{`/`}}`) in the file.

    Set reload=True to force a re-read from disk even if this template
    was already loaded once this process (handy when iterating on prompt
    wording without restarting the app).
    """
    template = _read_template(name, reload=reload)
    if not kwargs:
        return template
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(
            f"Prompt template '{name}' references placeholder {e} that "
            f"was not supplied. Provided keys: {sorted(kwargs.keys())}"
        )


def clear_prompt_cache():
    """Drops all cached templates so the next load_prompt() call re-reads from disk."""
    _cache.clear()
