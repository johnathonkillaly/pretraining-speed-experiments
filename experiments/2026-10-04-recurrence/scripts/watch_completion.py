"""Wait for append notifications, printing completed-fit events rather than polling trainers."""
import json
import select
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "metrics/suite_events.jsonl").open() as f:
        queue = select.kqueue()
        queue.control([select.kevent(f.fileno(), filter=select.KQ_FILTER_VNODE,
                                     flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                                     fflags=select.KQ_NOTE_WRITE)], 0, 0)
        while True:
            line = f.readline()
            if not line:
                queue.control(None, 1, 55)
                continue
            event = json.loads(line)
            if event["event"] == "run_end":
                tag = event["tag"]
                summary_path = ROOT / "metrics" / f"{tag}_summary.json"
                result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                print(json.dumps({"event": "completed_fit", "tag": tag, "exit_code": event["exit_code"],
                                  "development_loss": result.get("final_development_loss"),
                                  "test_loss": result.get("final_test_loss"),
                                  "steady_tokens_per_second": result.get("steady_tokens_per_second"),
                                  "training_seconds": result.get("training_seconds_including_compile")}), flush=True)
                if event["exit_code"] != 0:
                    raise RuntimeError(f"Controller reported failed fit: {tag}")
            elif event["event"] == "suite_complete":
                print(json.dumps(event), flush=True)
                return


if __name__ == "__main__":
    main()
