"""Create a fresh training slice and final-only research test documents."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "2026-10-04"
STORE = Path("/EXTERNAL_RESOURCE/56c91bc4ae0d")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    STORE.mkdir(exist_ok=False)
    for name in ("data", "checkpoints"):
        (STORE / name).mkdir()
        (ROOT / name).symlink_to(STORE / name, target_is_directory=True)
    packed = Path("/EXTERNAL_RESOURCE/9755c793eca0")
    manifest = json.loads((packed / "manifest.json").read_text())
    previous = [json.loads(line) for line in (OLD / "data/train_provenance.jsonl").read_text().splitlines()]
    excluded = {}
    for record in previous:
        excluded.setdefault(record["shard"], set()).add(record["block"])
    rng = np.random.default_rng(2026100417)
    train, provenance = [], []
    # The final shard is a short remainder; sample complete shards instead.
    chosen = np.linspace(0, len(manifest["shards"]) - 2, 16, dtype=int)
    for index in chosen:
        shard = manifest["shards"][int(index)]
        path = packed / shard["path"]
        assert sha(path) == shard["sha256"]
        blocks = np.memmap(path, dtype="<u2", mode="r").reshape(-1, 1024)
        available = np.array([i for i in range(len(blocks)) if i not in excluded.get(str(path), set())])
        assert len(available) >= 2500
        for block in rng.choice(available, 2500, replace=False):
            start = int(rng.integers(0, 768))
            train.append(blocks[int(block), start:start + 257].copy())
            provenance.append({"shard": str(path), "sha256": shard["sha256"],
                               "block": int(block), "start": start})
        print("Prepared shard", int(index), flush=True)
    order = rng.permutation(len(train))
    train = np.stack(train)[order]
    provenance = [provenance[int(i)] for i in order]
    np.save(STORE / "data/train.npy", train)
    (STORE / "data/train_provenance.jsonl").write_text("".join(json.dumps(row) + "\n" for row in provenance))
    shutil.copyfile(OLD / "data/validation.npy", STORE / "data/development.npy")
    tokenizer_path = Path(json.loads((OLD / "configs/data.json").read_text())["tokenizer_path"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    validation_path = tokenizer_path.parents[1] / "shards/val.jsonl"
    old_candidates = set(np.random.default_rng(999).choice(87335, 256, replace=False).tolist())
    available = np.array([i for i in range(87335) if i not in old_candidates])
    candidates = set(rng.choice(available, 2000, replace=False).tolist())
    test, test_provenance = [], []
    with validation_path.open() as file:
        for index, line in enumerate(file):
            if index not in candidates:
                continue
            record = json.loads(line)
            tokens = tokenizer.encode(record["text"]).ids[:1022] + [2]
            if len(tokens) < 257:
                continue
            test.append(np.array(tokens[:257], dtype=np.uint16))
            test_provenance.append({"line": index + 1, "dataset": record.get("_boothgpt_dataset_id"),
                                    "source": record.get("_boothgpt_source_path")})
    assert len(test) >= 512
    selected = rng.permutation(len(test))[:512]
    test = np.stack(test)[selected]
    test_provenance = [test_provenance[int(i)] for i in selected]
    development = np.load(STORE / "data/development.npy")
    train_hashes = {hashlib.sha256(row.tobytes()).hexdigest() for row in train}
    dev_hashes = {hashlib.sha256(row.tobytes()).hexdigest() for row in development}
    assert not any(hashlib.sha256(row.tobytes()).hexdigest() in train_hashes | dev_hashes for row in test)
    assert not any(hashlib.sha256(row.tobytes()).hexdigest() in train_hashes for row in development)
    for array in (train, development, test):
        assert array.max() < 32768
        assert (array[:, 1:] != 0).all()
    np.save(STORE / "data/test.npy", test)
    (STORE / "data/test_provenance.jsonl").write_text("".join(json.dumps(row) + "\n" for row in test_provenance))
    meta = {"training_shape": list(train.shape), "development_shape": list(development.shape),
            "test_shape": list(test.shape), "training_targets": int(train.shape[0] * 256),
            "seed": 2026100417, "training_shards": chosen.tolist(),
            "excluded_old_training_blocks": len(previous), "reused_old_blocks": 0,
            "exact_train_dev_test_window_overlap": 0,
            "test_dataset_counts": {name: sum(row["dataset"] == name for row in test_provenance)
                                    for name in {row["dataset"] for row in test_provenance}},
            "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": sha(tokenizer_path),
            "packed_manifest_sha256": sha(packed / "manifest.json"),
            "data_sha256": {name: sha(STORE / "data" / name) for name in
                            ("train.npy", "development.npy", "test.npy", "train_provenance.jsonl", "test_provenance.jsonl")},
            "limitations": "Original v7 train/val; old mix rather than current Fedora v2; no semantic deduplication"}
    (ROOT / "configs/data.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
