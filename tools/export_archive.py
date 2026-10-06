"""Build the public evidence archive from an existing local research checkout.

CPU/file processing only. Does not read datasets, checkpoints or Git history.
Historical benchmark scripts are archived, not executed.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    base = Path("research/moe-night/results")
    studies = {
        "2026-10-04", "2026-10-04-optimizers", "2026-10-04-recurrence",
        "2026-10-05-mlx-loss", "2026-10-05-mlx-batching", "2026-10-05-windows",
        "2026-10-05-h1-laptop", "2026-10-04-speed-options",
    }
    tracked = subprocess.check_output(["git", "-C", str(source), "ls-files"], text=True).splitlines()
    keep = []
    excluded_names = {
        "PREFLIGHT.json", "MAC_PREFLIGHT.json", "MAC_REFERENCE.json",
        "WINDOWS_WINDOW_SUMMARIES.json", "TRANSFER_VERIFICATION.json",
        "SOURCE_FREEZE.json", "RETURN_MANIFEST.json", "reference-tree-sha256.json",
        "GIT_HISTORY.txt", "RAW_RECORD_MANIFEST.json", "live.json",
        "original_git_status_before.txt", "original_git_status_after.txt",
        "background_policy_adjustment.json", "nice_policy_adjustment.json",
        "launchd_registration.json", "launcher_original_unloaded.json",
        "launcher_pause.json", "launcher_v2_registered.json", "launcher_cleanup.json",
        "freeze_audit.json", "isolation_audit.json", "preparation_failure.json",
        "reference_h1.yaml", "PLOT_ENVIRONMENT.json", "MAC_PREFLIGHT.json",
    }
    for name in tracked:
        p = Path(name)
        if len(p.parts) < 4 or p.parts[:3] != base.parts or p.parts[3] not in studies:
            continue
        if any(x in p.parts for x in ("source_references", "diagnostics", "healthcheck-metadata", "source")):
            continue
        if p.name in excluded_names or p.suffix not in {".md", ".py", ".json", ".jsonl", ".csv", ".txt", ".lock", ".png", ".m"}:
            continue
        if p.parts[3] == "2026-10-05-windows" and p.name == "README.md":
            continue
        if p.name == "environment.json" and p.parts[3] == "2026-10-04":
            continue  # contains a broad process inventory, not a public environment record
        if p.name.endswith(".sha256.txt"):
            continue
        keep.append(p)

    # Only known scientific plots, never arbitrary untracked artifacts.
    for study in ("2026-10-04", "2026-10-04-recurrence"):
        report = source / base / study / "REPORT.md"
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report.read_text()):
            p = Path(target) if target.startswith("/") else report.parent / target
            p = p.resolve()
            if p.suffix != ".png" or not p.is_relative_to((source / base / study / "plots").resolve()):
                raise RuntimeError("Unexpected plot target")
            relative = p.relative_to(source)
            if relative not in keep:
                keep.append(relative)

    selected = set(keep)
    metadata_keys = {
        "command", "command_line", "commandline", "processes", "compute_processes",
        "ps_initial_local_ml", "source_git_status", "git_status", "hostname",
        "host_name", "computer_name", "serial_number", "serialnumber",
        "mac_address", "ip_address", "primary_launcher", "pid", "pids",
    }
    path_pattern = re.compile(r"(?:/Users/|/Volumes/|/var/folders/|/private/var/)[^\s\"'`)\]}>,;]+|[A-Za-z]:[\\/]+Users[\\/]+[^\s\"'`)\]}>,;]+")
    private_ip = re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2,3}\b")

    def destination(p):
        return Path("experiments") / p.relative_to(base)

    def clean_string(s):
        s = s.replace(str(source / base), "/RESEARCH_ROOT/experiments")
        s = s.replace(str(source), "/RESEARCH_ROOT")
        def redact_path(match):
            return "/EXTERNAL_RESOURCE/" + sha(match.group(0).encode())[:12]
        s = path_pattern.sub(redact_path, s)
        s = private_ip.sub("PRIVATE_IP", s)
        s = re.sub(r"[A-Za-z0-9.-]*\.ghe\.com", "PRIVATE_HOST", s, flags=re.I)
        s = re.sub(r"\bw\d{2}_[A-Za-z0-9_]+\b", "WINDOWS_LAPTOP", s)
        return s

    def clean_json(v):
        if isinstance(v, dict):
            return {clean_string(k): clean_json(x) for k, x in v.items() if k.lower() not in metadata_keys}
        if isinstance(v, list):
            return [clean_json(x) for x in v]
        if isinstance(v, str):
            return clean_string(v)
        return v

    manifest = {"publication_date": "2026-10-06", "source_commit": subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "policy": "Selected evidence and historical sources; paths and process metadata sanitized. No datasets, tensor payloads, credentials, OS inventories or private Git history.",
        "files": []}
    for p in sorted(keep):
        original = (source / p).read_bytes()
        target = output / destination(p)
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".png":
            published = original
        elif p.suffix == ".json":
            value = clean_json(json.loads(original))
            published = (json.dumps(value, indent=2) + "\n").encode()
        elif p.suffix == ".jsonl":
            published = "".join(json.dumps(clean_json(json.loads(line))) + "\n"
                                for line in original.decode().splitlines() if line.strip()).encode()
        else:
            try:
                s = original.decode("utf-8")
            except UnicodeDecodeError:
                s = original.decode("cp1252")  # native-Windows contracts
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            if p == base / "2026-10-05-windows/returned/REPORT.md":
                s = re.sub(r"\n## Read-only laptop diagnostics\b.*?(?=\n## Provenance and deliverables)", "\n", s, flags=re.S)
            if p.suffix == ".md":
                def link(match):
                    label, url = match.groups()
                    if not url.startswith("/"):
                        return match.group(0)
                    q = Path(url)
                    if q.is_relative_to(source) and q.relative_to(source) in selected:
                        new = os.path.relpath(output / destination(q.relative_to(source)), target.parent)
                        return label + "(" + new + ")"
                    return label[1:-1] + " (local resource omitted)"
                s = re.sub(r"(\[[^\]]*\])\(([^)]+)\)", link, s)
                s = re.sub(r"\n{3,}", "\n\n", s)
                s = "> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.\n\n" + s
            published = (clean_string(s).rstrip() + "\n").encode()
        target.write_bytes(published)
        manifest["files"].append({"path": str(destination(p)), "source_path": str(p),
                                  "original_sha256": sha(original), "published_sha256": sha(published),
                                  "sanitized": original != published})

    original_environment = json.loads((source / base / "2026-10-04/configs/environment.json").read_text())
    public_environment = {k: original_environment[k] for k in ("recorded_at_local", "macos", "python", "packages", "device", "physical_memory_bytes")}
    envpath = output / "experiments/2026-10-04/configs/environment_public.json"
    envpath.write_text(json.dumps(public_environment, indent=2) + "\n")
    (output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"exported_files": len(manifest["files"]), "sanitized_files": sum(x["sanitized"] for x in manifest["files"]),
                      "exported_bytes": sum((output / x["path"]).stat().st_size for x in manifest["files"])}))


if __name__ == "__main__":
    main()
