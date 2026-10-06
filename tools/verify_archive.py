"""CPU-only publication integrity, format and batching-arithmetic verification."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    recorded = json.loads((ROOT / "PUBLICATION_SHA256.json").read_text())
    paths = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
             if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
             and p.name not in {"PUBLICATION_SHA256.json", ".DS_Store"}}
    assert paths == set(recorded), "Published tree differs from checksum manifest"
    for name, expected in recorded.items():
        assert digest(ROOT / name) == expected, name
    source_manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for row in source_manifest["files"]:
        assert digest(ROOT / row["path"]) == row["published_sha256"], row["path"]
    formats = 0
    for p in ROOT.rglob("*"):
        if ".git" in p.parts or not p.is_file():
            continue
        if p.suffix == ".json":
            json.loads(p.read_text())
            formats += 1
        elif p.suffix == ".jsonl":
            for line in p.read_text().splitlines():
                if line.strip():
                    json.loads(line)
            formats += 1

    study = ROOT / "experiments/2026-10-05-mlx-batching"
    summary = json.loads((study / "SUMMARY.json").read_text())
    runs = {}
    for batch in (1, 2, 4):
        measured = []
        for repeat in (1, 2):
            tag = f"B{batch}_R{repeat}"
            record = json.loads((study / "metrics" / (tag + ".json")).read_text())
            rows = [r for r in record["records"] if not r["warmup"]]
            assert len(rows) == 5
            measured.extend(rows)
            seconds = sum(r["total_seconds"] for r in rows)
            rate = len(rows) * 32760 / seconds
            runs[tag] = rate
            assert math.isclose(rate, summary["arms"][str(batch)]["per_run"][tag], rel_tol=1e-12)
        seconds = sum(r["total_seconds"] for r in measured)
        rate = len(measured) * 32760 / seconds
        arm = summary["arms"][str(batch)]
        assert math.isclose(rate, arm["targets_per_second"], rel_tol=1e-12)
        assert arm["numerical_gates_passed"] == (batch == 1)
    for batch in (2, 4):
        for pair in summary["decisions"][str(batch)]["pairs"]:
            repeat = pair["round"]
            gain = (runs[f"B{batch}_R{repeat}"] / runs[f"B1_R{repeat}"] - 1) * 100
            assert math.isclose(gain, pair["speed_gain_pct"], abs_tol=1e-10)
        assert not summary["decisions"][str(batch)]["useful"]
        assert not summary["decisions"][str(batch)]["strong"]
    print(f"PASS: {len(recorded)} publication hashes, {len(source_manifest['files'])} export hashes, {formats} JSON/JSONL files; six-run batching arithmetic and failed decisions preserved.")


if __name__ == "__main__":
    main()
