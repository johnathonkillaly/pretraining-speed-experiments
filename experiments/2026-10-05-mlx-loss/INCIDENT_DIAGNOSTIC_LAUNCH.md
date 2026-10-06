> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Diagnostic launcher preflight failure

2026-10-05. The first ephemeral diagnostic launcher tried to reuse the frozen
controller by replacing subprocess.Popen globally. That replacement also affected
its read-only `ps` preflight, creating an invalid command. Preflight failed before
any diagnostic GPU process started. No additional GPU budget was consumed.

The committed replacement launcher substitutes only the worker pathname in the
frozen controller definition prefix; it preserves normal subprocess behavior and
all safety-monitor/cleanup functions. It does not execute the controller's old
suite body. New runner and incident are hashed in DIAGNOSTIC_RUNNER_FREEZE.json
before execution. Original frozen files remain unchanged.
