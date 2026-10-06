"""Post-run descriptive analysis. Never changes frozen inputs or evaluates models."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("dense10", "dense4", "rec_full", "rec_tb2", "rec_curr_tb2")
SEEDS = (42, 43, 44)
LABELS = {
    "dense10": "Dense 10", "dense4": "Dense 4", "rec_full": "Recurrent, full",
    "rec_tb2": "Recurrent, last 2", "rec_curr_tb2": "Last 2 + curriculum",
}
COLORS = dict(zip(KINDS, ("#315c9b", "#68747e", "#b47924", "#16816c", "#ac527e")))


def read_rows(tag):
    return [json.loads(line) for line in (ROOT / "metrics" / f"{tag}.jsonl").read_text().splitlines()]


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.,
            "min": float(values.min()), "max": float(values.max()), "values": values.tolist()}


def paired_bootstrap(a, b, random):
    """Paired seeds and documents, resampled separately; exploratory with only 3 seeds."""
    differences = np.asarray(a) - np.asarray(b)
    draws = []
    for _ in range(4000):
        seeds = random.integers(0, differences.shape[0], size=differences.shape[0])
        docs = random.integers(0, differences.shape[1], size=differences.shape[1])
        draws.append(differences[np.ix_(seeds, docs)].mean())
    return {"mean_difference_nats": float(differences.mean()),
            "paired_seed_differences": differences.mean(axis=1).tolist(),
            "exploratory_bootstrap_95_interval": np.quantile(draws, (.025, .975)).tolist(),
            "resamples": 4000, "sampling": "paired seed and document axes separately; only three seeds"}


def phase_stats(rows, summary):
    train = [r for r in rows if r["event"] == "train"]
    signatures = {}
    for row in train:
        signatures.setdefault((row["loops"], row["backprop_loops"]), []).append(row)
    paths = []
    for (loops, k), calls in signatures.items():
        steady = [r["step_ms"] for r in calls if r["steady"]]
        timing = describe(steady)
        timing.pop("values")
        timing["median"] = float(np.median(steady))
        paths.append({"loops": loops, "backprop_loops": k, "calls": len(calls),
                      "first_call_ms_including_compile": calls[0]["step_ms"],
                      "steady_calls": len(steady), "steady_step_ms": timing,
                      "steady_tokens_per_second": 2048000 / np.mean(steady)})
    validation = rows[0]["elapsed_seconds"]
    test = 0.
    for previous, current in zip(rows, rows[1:]):
        if current["event"] == "validation":
            validation += current["elapsed_seconds"] - previous["elapsed_seconds"]
        elif current["event"] == "final_test":
            test += current["elapsed_seconds"] - previous["elapsed_seconds"]
    norms = np.array([r["gradient_norm"] for r in train])
    return {"signatures": paths, "initial_validation_seconds": rows[0]["elapsed_seconds"],
            "gradient_norm_mean": float(norms.mean()), "gradient_norm_median": float(np.median(norms)),
            "gradient_norm_p95": float(np.quantile(norms, .95)), "fraction_gradient_norm_above_one": float(np.mean(norms > 1)),
            "development_validation_seconds": validation, "final_test_seconds": test,
            "checkpoint_seconds": summary["checkpoint_seconds"],
            "remaining_python_data_logging_seconds": summary["elapsed_seconds"]
            - summary["training_seconds_including_compile"] - validation - test - summary["checkpoint_seconds"]}


def curve(rows, axis):
    vals = [r for r in rows if r["event"] == "validation"]
    x = np.array([r[axis] for r in vals]); y = np.array([r["loss"] for r in vals])
    assert np.all(np.diff(x) > 0)
    return x, y


def plot_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5), constrained_layout=True)
    for kind in KINDS:
        for ax, axis, scale in zip(axes, ("tokens", "training_seconds"), (1e6, 60.)):
            curves = [curve(histories[kind, seed], axis) for seed in SEEDS]
            limit = min(x[-1] for x, _ in curves)
            grid = np.linspace(0, limit, 180)
            loss = np.array([np.interp(grid, x, y) for x, y in curves])
            mean, sd = loss.mean(axis=0), loss.std(axis=0, ddof=1)
            ax.plot(grid / scale, mean, label=LABELS[kind], color=COLORS[kind], lw=2)
            ax.fill_between(grid / scale, mean - sd, mean + sd, color=COLORS[kind], alpha=.13)
    axes[0].set(xlabel="Prediction targets seen (millions)", ylabel="Development loss (nats/token)",
                title="Same token budget")
    axes[1].set(xlabel="Measured training minutes, including compile", ylabel="Development loss (nats/token)",
                title="Learning per measured training time")
    axes[1].legend(fontsize=9, loc="upper right")
    for ax in axes:
        ax.grid(alpha=.2); ax.set_ylim(bottom=5.5)
    fig.suptitle("Native pretraining pilot · three seeds · mean ± seed SD", fontsize=13)
    fig.savefig(ROOT / "plots" / "learning_curves.png", dpi=180); plt.close(fig)


def plot_final(table, summaries):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    x = np.arange(len(KINDS))
    for i, kind in enumerate(KINDS):
        loss = table[kind]["final_test_loss"]
        axes[0].errorbar(i, loss["mean"], yerr=loss["sample_sd"], fmt="o", capsize=5,
                         color=COLORS[kind], markersize=8)
        axes[0].scatter(i + np.array([-.09, 0, .09]), loss["values"], color=COLORS[kind], s=16, alpha=.65)
        speed = table[kind]["steady_tokens_per_second"]
        axes[1].bar(i, speed["mean"] / 1000, color=COLORS[kind], alpha=.75,
                    yerr=speed["sample_sd"] / 1000, capsize=5)
    for ax in axes:
        ax.set_xticks(x, [LABELS[k].replace(", ", "\n").replace(" + ", "\n+ ") for k in KINDS], fontsize=9)
        ax.grid(axis="y", alpha=.2)
    axes[0].set(title="Final-only test · 512 documents", ylabel="Loss (nats/token), lower is better")
    axes[1].set(title="Compiled complete training update", ylabel="Steady throughput (thousands of targets/s)")
    fig.suptitle("10.24M prediction targets per fit · points/error bars show three seeds", fontsize=12)
    fig.savefig(ROOT / "plots" / "final_comparison.png", dpi=180); plt.close(fig)


def plot_backward(backward):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    x = np.arange(len(backward)); labels = [LABELS[r["kind"]].replace(", ", "\n") for r in backward]
    forwards = [r["median"]["forward_ms"] for r in backward]
    backwards = [r["median"]["backward_ms"] for r in backward]
    axes[0].bar(x, forwards, label="Forward", color="#a4bed4")
    axes[0].bar(x, backwards, bottom=forwards, label="Backward", color="#386b95")
    axes[0].set(ylabel="Milliseconds", title="Separated eager arithmetic"); axes[0].legend(fontsize=9)
    axes[1].bar(x, [r["median"]["peak_incremental_bytes"] / 2**30 for r in backward], color="#607ea2")
    axes[1].set(ylabel="GiB above model/input baseline", title="Peak forward/backward allocation")
    axes[2].bar(x, [r["median"]["gradient_physical_retained_bytes"] / 2**20 for r in backward], color="#6a9386")
    axes[2].set(ylabel="MiB", title="Median retained gradient storage")
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=9); ax.grid(axis="y", alpha=.2)
    fig.suptitle("BF16 · batch8 × context256 · median of 15 windows after 4 initial calls", fontsize=12)
    fig.savefig(ROOT / "plots" / "backward_cost.png", dpi=180); plt.close(fig)


def plot_hardware(h1):
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    fig.subplots_adjust(left=.37, right=.96, bottom=.23, top=.89)
    values = [h1["mean_tokens_per_second"], 15522.91]
    bars = ax.barh(["M4 Max / MLX\nnew synthetic probe", "RTX 3080 / CUDA\nSeptember 90-update record"],
                   np.array(values) / 1000, color=["#687b9a", "#459075"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + .2, bar.get_y() + bar.get_height()/2, f"{value:,.0f}", va="center")
    ax.set(xlim=(0, 18), xlabel="Prediction targets/s (thousands)",
           title="H1-shaped practical pipeline context · 298.36M parameters")
    ax.grid(axis="x", alpha=.2)
    fig.text(.5, .04, "Same shape and update token batch; different code/data/date.\nNo power normalization or hardware-ceiling claim.",
             ha="center", fontsize=8)
    fig.savefig(ROOT / "plots" / "hardware_context.png", dpi=180); plt.close(fig)


def main():
    assert (ROOT / "metrics/SUITE_COMPLETE.json").exists(), "Wait for all primary fits"
    for freeze_file in ("FREEZE.json", "H1_V2_FREEZE.json", "PRIORITY_FREEZE.json"):
        for filename, digest in json.loads((ROOT / freeze_file).read_text())["sha256"].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
    events = [json.loads(line) for line in (ROOT / "metrics/suite_events.jsonl").read_text().splitlines()]
    starts = {r["tag"]: datetime.datetime.fromisoformat(r["utc"]) for r in events if r["event"] == "run_start"}
    ends = {r["tag"]: datetime.datetime.fromisoformat(r["utc"]) for r in events if r["event"] == "run_end"}
    summaries, histories, tests = {}, {}, {}
    phases = {}
    for kind in KINDS:
        for seed in SEEDS:
            tag = f"{kind}_s{seed}"
            s = json.loads((ROOT / "metrics" / f"{tag}_summary.json").read_text())
            s["process_elapsed_seconds"] = (ends[tag] - starts[tag]).total_seconds()
            rows = read_rows(tag)
            train = [r for r in rows if r["event"] == "train"]
            test = [r for r in rows if r["event"] == "final_test"]
            assert len(train) == 5000 and [r["step"] for r in train] == list(range(1, 5001))
            assert len(test) == 1 and len(test[0]["per_document_loss"]) == 512
            assert s["steps"] == 5000 and not s["tune"]
            assert all(np.isfinite(r["loss"]) and np.isfinite(r["gradient_norm"]) for r in train)
            assert train[-1]["tokens"] == 10_240_000
            assert abs(np.mean(test[0]["per_document_loss"]) - s["final_test_loss"]) < 1e-10
            summaries[kind, seed] = s; histories[kind, seed] = rows
            tests[kind, seed] = test[0]["per_document_loss"]
            phases[tag] = phase_stats(rows, s)
    for seed in SEEDS:
        assert len({summaries[kind, seed]["order_sha256"] for kind in KINDS}) == 1
        assert len({summaries[kind, seed]["shared_initialization_sha256"] for kind in KINDS}) == 1
    table = {}
    for kind in KINDS:
        ss = [summaries[kind, seed] for seed in SEEDS]
        table[kind] = {"parameter_count": ss[0]["parameter_count"],
                       "muon_peak": ss[0]["muon_peak"], "adamw_peak": ss[0]["adamw_peak"]}
        for field in ("final_test_loss", "final_development_loss", "steady_tokens_per_second",
                      "training_seconds_including_compile", "elapsed_seconds", "process_elapsed_seconds", "peak_active_bytes"):
            table[kind][field] = describe([s[field] for s in ss])
    comparisons = {}
    random = np.random.default_rng(2026100418)
    for candidate, reference in (("rec_tb2", "rec_full"), ("rec_tb2", "dense10"),
                                 ("rec_tb2", "dense4"), ("rec_curr_tb2", "rec_tb2"),
                                 ("rec_full", "dense10")):
        row = paired_bootstrap([tests[candidate, seed] for seed in SEEDS],
                               [tests[reference, seed] for seed in SEEDS], random)
        row["paired_training_time_ratios"] = [summaries[candidate, seed]["training_seconds_including_compile"]
                                               / summaries[reference, seed]["training_seconds_including_compile"] for seed in SEEDS]
        row["paired_steady_throughput_ratios"] = [summaries[candidate, seed]["steady_tokens_per_second"]
                                                  / summaries[reference, seed]["steady_tokens_per_second"] for seed in SEEDS]
        comparisons[f"{candidate}_minus_{reference}"] = row
    # Descriptive interpolation of registered dev curves, not another fitted/evaluated endpoint.
    equal_wall = {"method": "linear interpolation between dev evaluations; includes first-call compilation; excludes validation/data/checkpoint time",
                  "limitation": "Schedules cool by token count; these are fixed-token runs sampled at equal compute time, not equal-wall-budget fits",
                  "per_seed": {}}
    equal_elapsed = {"method": "linear interpolation of dev checkpoints against elapsed fit time including initial/intermediate validation and Python overhead",
                     "per_seed": {}}
    for seed in SEEDS:
        limit = min(summaries[kind, seed]["training_seconds_including_compile"] for kind in KINDS)
        vals = {}
        for kind in KINDS:
            x, y = curve(histories[kind, seed], "training_seconds")
            vals[kind] = float(np.interp(limit, x, y))
        equal_wall["per_seed"][str(seed)] = {"seconds": limit, "development_loss": vals}
        elapsed_limit = min(curve(histories[kind, seed], "elapsed_seconds")[0][-1] for kind in KINDS)
        vals = {kind: float(np.interp(elapsed_limit, *curve(histories[kind, seed], "elapsed_seconds"))) for kind in KINDS}
        equal_elapsed["per_seed"][str(seed)] = {"seconds": elapsed_limit, "development_loss": vals}
    for output in (equal_wall, equal_elapsed):
        output["aggregate_development_loss"] = {kind: describe([r["development_loss"][kind] for r in output["per_seed"].values()]) for kind in KINDS}
    provenance = [json.loads(s) for s in (ROOT / "data/test_provenance.jsonl").read_text().splitlines()]
    source_field = "dataset" if "dataset" in provenance[0] else "source"
    domains = {}
    for source in sorted({r[source_field] for r in provenance}):
        indices = [i for i, r in enumerate(provenance) if r[source_field] == source]
        domains[source] = {"documents": len(indices), "loss": {kind: describe([np.asarray(tests[kind, seed])[indices].mean() for seed in SEEDS]) for kind in KINDS}}
    backward = json.loads((ROOT / "metrics/backward_benchmark.json").read_text())
    h1 = json.loads((ROOT / "metrics/h1_benchmark_v2.json").read_text())
    # Explanatory projection-only proxy, not a profiler measurement or decision gate.
    block_matrices = 2 * 256**2 + 2 * 256 * 64 + 3 * 256 * 864
    head_matrix = 32768 * 256
    flop_proxy = {"scope": "linear projections only; excludes attention score/value products, softmax/CE, norms, activation/injection, optimizer and runtime costs",
                  "block_matrix_parameters": block_matrices, "head_matrix_parameters": head_matrix,
                  "assumption": "2 FLOPs per multiply-add; backward input+weight products cost twice forward projection",
                  "per_prediction_target": {}}
    for kind, forward_blocks, backward_blocks in (("dense10", 10, 10), ("dense4", 4, 4),
                                                   ("rec_full", 10, 10), ("rec_tb2", 10, 6),
                                                   ("rec_curr_tb2", 9.5, 5.9)):
        forward = 2 * (forward_blocks * block_matrices + head_matrix)
        backward_flops = 4 * (backward_blocks * block_matrices + head_matrix)
        flop_proxy["per_prediction_target"][kind] = {"mean_executed_forward_blocks": forward_blocks,
                                                     "mean_differentiated_blocks": backward_blocks,
                                                     "forward_flops": forward, "backward_flops": backward_flops,
                                                     "total_flops": forward + backward_flops}
    output = {"primary_fits": 15, "targets_per_fit": 10_240_000, "seed_sd_is_not_ci": True,
              "summary": table, "paired_test_comparisons_exploratory": comparisons,
              "equal_training_time_development": equal_wall, "equal_elapsed_time_development": equal_elapsed,
              "test_source_profile": domains, "fit_phase_times": phases,
              "backward_medians": {r["kind"]: r["median"] for r in backward},
              "projection_flop_proxy_explanatory": flop_proxy,
              "primary_process_seconds_total": sum(s["process_elapsed_seconds"] for s in summaries.values()),
              "primary_training_seconds_total": sum(s["training_seconds_including_compile"] for s in summaries.values()),
              "h1_context": {"mlx_tokens_per_second": h1["mean_tokens_per_second"], "cuda_historical_tokens_per_second": 15522.91,
                             "historical_pipeline_ratio": 15522.91 / h1["mean_tokens_per_second"],
                             "not_controlled_hardware_comparison": True}}
    (ROOT / "metrics/analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    with (ROOT / "metrics/primary_summary.csv").open("w", newline="") as f:
        fields = ["kind", "seed", "parameter_count", "muon_peak", "final_test_loss", "final_development_loss",
                  "steady_tokens_per_second", "training_seconds_including_compile", "elapsed_seconds", "process_elapsed_seconds", "peak_active_bytes"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for kind in KINDS:
            for seed in SEEDS:
                writer.writerow({key: summaries[kind, seed][key] for key in fields})
    (ROOT / "plots").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    plot_curves(histories); plot_final(table, summaries); plot_backward(backward); plot_hardware(h1)
    print(json.dumps({"summary": table, "paired_comparisons": comparisons,
                      "equal_time": equal_wall["aggregate_development_loss"]}, indent=2))


if __name__ == "__main__":
    main()
