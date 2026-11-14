"""Local file helpers; no remote transport or credentials."""
from pathlib import Path
import hashlib
import json

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()

def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
