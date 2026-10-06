> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Numerical and capacity outcomes before timings

October 5, 2026, America/Denver. Registered loss and gradient tolerances are unchanged.

| Context inputs | Physical batch | Loss absolute difference | Gradient relative L2 | Registered numerical gate |
|---|---:|---:|---:|---|
| 64 | 1 | 0 | 0 | PASS |
| 64 | 2 | 0.00035727024 | 0.01179004 | FAIL |
| 64 | 4 | 0.00035703182 | 0.011825139 | FAIL |
| 64 | 8 | 0.00035703182 | 0.011613248 | FAIL |
| 4095 | 1 | 0 | 0 | PASS |
| 4095 | 2 | 4.7683716e-07 | 0.0023510628 | PASS |
| 4095 | 4 | 2.3841858e-07 | 0.0023634547 | PASS |

Every checked parameter gradient was finite and nonzero. B2/B4 fail the short-context gate but pass the actual 4095-position gate. Their timings remain diagnostic because the contract requires both. This failure is numerical non-equivalence under a strict tolerance; no real-data learning degradation has been measured, and no explanation is established by these checks alone.

B8 passed neither short-context threshold. It was skipped at full context: the preregistered forecast from B2/B4 is 94.97GiB, exceeding the 76GiB forecast threshold inside the 80GiB allocation budget. This is a forecast-based capacity skip, not an observed batch-8 OOM or an actual batch-8 peak.

Correctness peaks retain the B1 reference gradient and are not the headline timing-memory peaks. No threshold was relaxed after seeing results.
