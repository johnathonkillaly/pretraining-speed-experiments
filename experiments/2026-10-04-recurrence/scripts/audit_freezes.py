"""Read-only hashes across this study and its earlier frozen dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    for directory in (ROOT, ROOT.parent / "2026-10-04", ROOT.parent / "2026-10-04-optimizers"):
        for freeze in sorted(directory.glob("*FREEZE.json")):
            data = json.loads(freeze.read_text())
            mapping = data.get("sha256", data.get("files"))
            assert isinstance(mapping, dict), freeze
            for filename, expected in mapping.items():
                path = Path(filename)
                if not path.is_absolute():
                    path = directory / path
                if isinstance(expected, dict):
                    expected = expected["sha256"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
            records.append({"freeze": str(freeze), "files": len(mapping), "passed": True})
    (ROOT / "metrics/freeze_audit.json").write_text(json.dumps({"passed": True, "records": records}, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
