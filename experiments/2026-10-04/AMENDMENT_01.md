> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Pretraining smoke amendment, 2026-10-04 MDT

Initial microbenchmark completed before training. Its allocator delta included
the evaluated loss object, which retained its forward graph. A standalone
deletion probe exposed that confound. Add a gradient-only delta after releasing
the loss, retaining every gradient array. Keep original measurements as v1
evidence, rerun into a new v2 file. Also export graphs before evaluation (graphs
exported after evaluation contain leaves only). Timing is unchanged; graph
exports occur outside timed windows. These are instrumentation repairs,
not changes to decision rules or training configurations.
