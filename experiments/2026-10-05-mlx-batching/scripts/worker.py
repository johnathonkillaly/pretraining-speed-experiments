"""Frozen physical-batch comparison. Synthetic inputs; no tensor payloads saved."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PREVIOUS = STUDY.parent / "2026-10-05-mlx-loss"


def verify_freezes():
    files = [STUDY / "FREEZE.json", *sorted(PREVIOUS.glob("*FREEZE.json"))]
    for path in files:
        for filename, digest in json.loads(path.read_text())["sha256"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Freeze mismatch: {filename}")


verify_freezes()
spec = importlib.util.spec_from_file_location("_frozen_loss_worker", PREVIOUS / "scripts/worker.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
mx, nn, np = reference.mx, reference.nn, reference.np
assert mx.__version__ == "0.32.3" and np.__version__ == "2.4.6"
assert sys.version.startswith("3.11.17")
tree_flatten = reference.tree_flatten
save = reference.save
GIB = 1024**3
mx.set_memory_limit(80 * GIB)
mx.set_cache_limit(GIB)


def log(**row):
    print(json.dumps(row), flush=True)


def guard(**phase):
    record = {"active_bytes": mx.get_active_memory(), "peak_active_bytes": mx.get_peak_memory(),
              "cache_bytes": mx.get_cache_memory(), **phase}
    save(STUDY / "metrics/live.json", record)
    if record["peak_active_bytes"] > 80 * GIB:
        raise RuntimeError(f"Active allocation stop: {record['peak_active_bytes']}")


def master_hash(trainer):
    digest = hashlib.sha256()
    for name in trainer.optimizer.names:
        digest.update(name.encode())
        digest.update(np.array(trainer.optimizer.state["masters"][name]).tobytes())
    return digest.hexdigest()


def prepare(context):
    started = time.perf_counter()
    trainer = reference.Trainer("dense20", 42, reference.args_h1(), checkpoint_mode="none")
    count = sum(p.size for _, p in tree_flatten(trainer.model.parameters()))
    assert count == 298361856
    initial_hash = master_hash(trainer)
    full = np.concatenate([np.random.default_rng(500+i).integers(1, 32768, size=(1, 4096)).astype(np.int32)
                           for i in range(8)], axis=0)
    identity = {"parameters": count, "initialization_sha256": initial_hash,
                "full_input_sha256": hashlib.sha256(full.tobytes()).hexdigest(),
                "input_sha256": hashlib.sha256(full[:, :context+1].tobytes()).hexdigest(),
                "context_inputs": context, "targets_per_update": 8 * context,
                "initialization_seconds": time.perf_counter()-started,
                "mlx": mx.__version__, "loss": "composed_fp32", "checkpointing": "none",
                "dtype": "BF16 model/backward; FP32 masters/accumulation/state",
                "memory_allowance_bytes": 80*GIB, "cache_allowance_bytes": GIB}
    return trainer, full[:, :context+1], identity


def functions(trainer):
    value_grad = nn.value_and_grad(trainer.model,
        lambda x, y: reference.model_objective(trainer.model, "composed", x, y))
    def gradient(x, y):
        loss, grads = value_grad(x, y)
        return loss, {name: value.astype(mx.float32) for name, value in tree_flatten(grads)}
    return mx.compile(gradient, inputs=trainer.model.state, outputs=trainer.model.state)


def batches(rows, size):
    result = [mx.array(rows[i:i+size]) for i in range(0, 8, size)]
    mx.eval(result)
    return result


def accumulate(fn, groups, tag):
    k = len(groups)
    accumulated, losses = None, []
    for index, batch in enumerate(groups):
        loss, grad = fn(batch[:, :-1], batch[:, 1:])
        accumulated = ({name: value/k for name, value in grad.items()} if accumulated is None else
                       {name: accumulated[name]+value/k for name, value in grad.items()})
        mx.eval(loss, accumulated)
        losses.append(float(loss))
        guard(tag=tag, microbatch_index=index)
        del loss, grad
    return float(np.mean(losses)), accumulated


def compare(a, b):
    errors = {name: reference.tensor_errors(value, b[name]) for name, value in a.items()}
    candidate_norms = [mx.sum(value*value) for value in b.values()]
    mx.eval(candidate_norms)
    relative = math.sqrt(sum(v["squared_difference"] for v in errors.values()) /
                         max(sum(v["reference_squared_norm"] for v in errors.values()), 1e-30))
    return {"relative_l2": relative, "finite": all(v["finite"] for v in errors.values()),
            "reference_nonzero": all(v["reference_squared_norm"] > 0 for v in errors.values()),
            "candidate_nonzero": all(float(v) > 0 for v in candidate_norms), "parameter_errors": errors}


def validation(context):
    trainer, rows, identity = prepare(context)
    fn = functions(trainer)
    report = {**identity, "stage": "validation", "arms": {}, "thresholds": {"loss_abs": 2e-4, "relative_l2": .005}}
    destination = STUDY / "metrics" / f"validation_{context}.json"
    baseline_loss, baseline = None, None
    for size in (1, 2, 4, 8):
        if context == 4095 and size == 8:
            p2 = report["arms"]["2"]["peak_active_bytes"]
            p4 = report["arms"]["4"]["peak_active_bytes"]
            forecast = p4+2*max(0, p4-p2)
            if forecast > 76*GIB:
                report["arms"]["8"] = {"status": "capacity_skipped", "forecast_active_bytes": forecast,
                                        "forecast_limit_bytes": 76*GIB}
                save(destination, report)
                log(event="capacity_skip", context=context, batch_size=size, forecast_active_bytes=forecast)
                continue
        groups = batches(rows, size)
        mx.reset_peak_memory()
        start = time.perf_counter()
        loss, gradients = accumulate(fn, groups, f"gate-{context}-B{size}")
        peak = mx.get_peak_memory()
        if baseline is None:
            baseline_loss, baseline = loss, gradients
        errors = compare(baseline, gradients)
        errors["loss_abs"] = abs(loss-baseline_loss)
        finite = errors["finite"] and errors["reference_nonzero"] and errors["candidate_nonzero"]
        passed = finite and errors["loss_abs"] <= 2e-4 and errors["relative_l2"] <= .005
        arm = {"batch_size": size, "accumulation": 8//size, "status": "checked", "loss": loss,
               "seconds": time.perf_counter()-start, "peak_active_bytes": peak, "passed": passed, **errors}
        report["arms"][str(size)] = arm
        save(destination, report)
        log(event="gradient_check", context=context, batch_size=size, passed=passed,
            relative_l2=errors["relative_l2"], loss_abs=errors["loss_abs"], peak_active_bytes=peak)
        if not finite:
            raise RuntimeError("Nonfinite or zero parameter gradient")
        if size != 1:
            del gradients
        del groups
        gc.collect()
        mx.clear_cache()
    report["completed"] = True
    save(destination, report)
    log(event="validation_complete", context=context)


def timing(size, tag):
    trainer, rows, identity = prepare(4095)
    fn = functions(trainer)
    groups = batches(rows, size)
    update = mx.compile(trainer.optimizer.apply, inputs=trainer.capture, outputs=trainer.capture)
    rates = mx.array(.00015), mx.array(.00006)
    mx.eval(rates)
    gate_records = [json.loads((STUDY/"metrics"/f"validation_{c}.json").read_text())["arms"][str(size)]
                    for c in (64, 4095)]
    numerical_passed = all(a.get("passed", False) for a in gate_records)
    report = {**identity, "stage": "timing", "tag": tag, "batch_size": size,
              "accumulation": 8//size, "numerical_gates_passed": numerical_passed,
              "diagnostic_only": not numerical_passed, "records": []}
    destination = STUDY/"metrics"/(tag+".json")
    for index in range(8):
        mx.reset_peak_memory()
        start = time.perf_counter()
        loss, accumulated = accumulate(fn, groups, tag)
        backward_end = time.perf_counter()
        if index == 0:
            assert reference.finite_tree(accumulated)
        optimizer_start = time.perf_counter()
        norm = update(accumulated, *rates)
        mx.eval(norm, trainer.capture)
        end = time.perf_counter()
        guard(tag=tag, update_index=index)
        assert math.isfinite(loss) and math.isfinite(float(norm))
        model_seconds = backward_end-start
        optimizer_seconds = end-optimizer_start
        row = {"index": index, "warmup": index < 3,
               "model_backward_accumulation_seconds": model_seconds, "optimizer_seconds": optimizer_seconds,
               "total_seconds": model_seconds+optimizer_seconds, "mean_loss": loss,
               "gradient_norm": float(norm), "peak_active_bytes": mx.get_peak_memory(),
               "active_bytes": mx.get_active_memory(), "cache_bytes": mx.get_cache_memory()}
        row["targets_per_second"] = 32760/row["total_seconds"]
        if index == 0:
            after = master_hash(trainer)
            assert after != identity["initialization_sha256"]
            report["first_update_master_sha256"] = after
            report["finite_first_state"] = reference.finite_tree(trainer.capture)
            assert report["finite_first_state"]
        report["records"].append(row)
        save(destination, report)
        log(event="update", tag=tag, **row)
        del accumulated, norm
    report["finite_final_state"] = reference.finite_tree(trainer.capture)
    assert report["finite_final_state"]
    report["steady_targets_per_second"] = 32760/np.mean([r["total_seconds"] for r in report["records"][3:]])
    report["peak_active_bytes"] = max(r["peak_active_bytes"] for r in report["records"])
    report["completed"] = True
    save(destination, report)
    log(event="timing_complete", tag=tag, targets_per_second=report["steady_targets_per_second"],
        peak_active_bytes=report["peak_active_bytes"], diagnostic_only=report["diagnostic_only"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["validation", "timing"])
    p.add_argument("--context", type=int, choices=[64, 4095], default=4095)
    p.add_argument("--batch", type=int, choices=[1, 2, 4, 8], default=1)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    try:
        validation(args.context) if args.stage == "validation" else timing(args.batch, args.tag)
        verify_freezes()
    except BaseException as error:
        save(STUDY/"metrics"/(args.tag+"_failure.json"), {"error": repr(error)})
        raise
