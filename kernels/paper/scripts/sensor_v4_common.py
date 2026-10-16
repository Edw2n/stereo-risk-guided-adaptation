"""Shared, versioned utilities. Never silently overwrite a run or data asset."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/sensor_selection_v4.json'


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


def seed(*parts):
    return int(digest(parts)[:16], 16)


def read_config(path=DEFAULT_CONFIG):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def storage(config, *parts):
    base = Path(config['storage_root']).expanduser().resolve()
    guard = Path(config.get('storage_guard_root', base)).expanduser().resolve()
    out = base.joinpath(*parts).resolve()
    if not base.is_relative_to(guard) or not out.is_relative_to(base):
        raise ValueError('Artifact escapes configured storage root/guard')
    out.mkdir(parents=True, exist_ok=True)
    return out


def require_approval(path, config, manifest):
    if not path:
        raise PermissionError('HARD GATE: full inference needs user GO and matching approval artifact')
    approval = json.loads(Path(path).read_text())
    if not (approval.get('user_decision') == 'GO' and approval.get('phase') == 'A'
            and approval.get('config_sha256') == digest(config)
            and approval.get('manifest_sha256') == digest(manifest)):
        raise PermissionError('Approval does not match this Phase A config/manifest')


def assert_dev(records):
    if not records or any(r['split'] != 'dev' for r in records):
        raise PermissionError('This operation accepts dev records only')
