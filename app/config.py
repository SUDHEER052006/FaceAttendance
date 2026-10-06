"""Config loading. Single source of truth = config.yaml at the repo root."""
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.environ.get("FA_CONFIG", os.path.join(ROOT, "config.yaml"))

_cache = {"mtime": None, "data": {}}


def load(force=False):
    """Reload config.yaml if it changed on disk. Cheap enough to call often."""
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        mtime = None
    if force or mtime != _cache["mtime"]:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            _cache["data"] = yaml.safe_load(fh) or {}
        _cache["mtime"] = mtime
    return _cache["data"]


def g(path, default=None):
    """Dotted lookup: g('detect.score_threshold', 0.75)."""
    node = load()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node if node is not None else default


def abspath(path):
    """Resolve a config path relative to the repo root."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def save(data):
    """Write config back (used by the Settings page)."""
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    os.replace(tmp, CONFIG_PATH)
    load(force=True)
