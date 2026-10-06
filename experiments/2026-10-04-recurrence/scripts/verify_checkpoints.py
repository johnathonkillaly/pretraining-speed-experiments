"""CPU-only checkpoint integrity checks; does not initialize MLX or touch original weights."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("dense10", "dense4", "rec_full", "rec_tb2", "rec_curr_tb2")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def check(tag):
    directory = ROOT / "checkpoints" / tag
    summary = json.loads((ROOT / "metrics" / f"{tag}_summary.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata == {"step": 5000, "tokens_seen": 10_240_000}
    path = directory / "model.safetensors"
    with path.open("rb") as f:
        header_bytes = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_bytes))
    header.pop("__metadata__", None)
    offsets = sorted(row["data_offsets"] for row in header.values())
    assert offsets[0][0] == 0 and all(a[1] == b[0] for a, b in zip(offsets, offsets[1:]))
    assert path.stat().st_size == 8 + header_bytes + offsets[-1][1]
    assert sum(np.prod(row["shape"]) for row in header.values()) == summary["parameter_count"]
    names = set(header)
    with np.load(directory / "training_state.npz", allow_pickle=False) as state:
        assert int(state["step"]) == 5000 and state["step"].dtype == np.uint32
        masters = {name.removeprefix("masters.") for name in state.files if name.startswith("masters.")}
        assert masters == names
        matrix_names = set()
        adam_names = set()
        for name, row in header.items():
            assert row["dtype"] == "BF16", (name, row["dtype"])
            first, last = row["data_offsets"]
            assert last - first == 2 * np.prod(row["shape"])
            bits = np.memmap(path, dtype="<u2", mode="r", offset=8 + header_bytes + first, shape=tuple(row["shape"]))
            assert np.all((bits & np.uint16(0x7f80)) != np.uint16(0x7f80)), name
            master = state["masters." + name]
            assert master.dtype == np.float32 and master.shape == bits.shape and np.isfinite(master).all()
            # BF16 round-to-nearest, ties-to-even: ensure saved model corresponds to final masters.
            raw = master.view(np.uint32)
            rounded = ((raw + np.uint32(0x7fff) + ((raw >> 16) & np.uint32(1))) >> 16).astype(np.uint16)
            assert np.array_equal(bits, rounded), f"Stale weights or unexpected BF16 conversion: {tag}/{name}"
            if "muon_m." + name in state.files:
                matrix_names.add(name)
            else:
                adam_names.add(name)
            del bits, master, raw, rounded
        expected = {"step"} | {"masters." + name for name in names} | {"muon_m." + name for name in matrix_names}
        expected |= {branch + "." + name for branch in ("adam_m", "adam_v") for name in adam_names}
        assert set(state.files) == expected
        assert len(matrix_names) == summary["optimizer_ownership"]["muon_tensors"]
        assert len(adam_names) == summary["optimizer_ownership"]["adamw_tensors"]
        for name in state.files:
            value = state[name]
            assert np.isfinite(value).all(), name
            if name != "step":
                assert value.dtype == np.float32
                parameter_name = name.partition(".")[2]
                assert list(value.shape) == header[parameter_name]["shape"]
    return {"tag": tag, "passed": True, "step": 5000, "tokens_seen": 10_240_000,
            "parameters": summary["parameter_count"], "weight_tensors": len(names),
            "finite_weights_and_state": True, "BF16_weights_exactly_match_rounded_FP32_masters": True,
            "files": {name: {"bytes": (directory / name).stat().st_size,
                              "sha256": sha256(directory / name)} for name in ("model.safetensors", "training_state.npz", "metadata.json")}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--tag"); args = parser.parse_args()
    tags = [args.tag] if args.tag else [f"{kind}_s{seed}" for seed in (42, 43, 44) for kind in KINDS]
    records = []
    for tag in tags:
        record = check(tag); records.append(record)
        print(json.dumps({"tag": tag, "passed": True, "parameters": record["parameters"]}), flush=True)
    name = f"checkpoint_check_{args.tag}.json" if args.tag else "checkpoint_checks.json"
    (ROOT / "metrics" / name).write_text(json.dumps({"passed": True, "CPU_only": True, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
