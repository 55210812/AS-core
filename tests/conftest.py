"""Small JSON fixture writer; independent of any experiment manager."""
import json


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
