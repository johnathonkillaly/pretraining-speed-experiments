> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Post-probe amendment: stateless SGD control

The initial 32-case probe completed successfully. An unexpected implementation
detail merits one additional control: MLX 0.32.2 SGD.init_single allocates a full
velocity buffer even when momentum=0, although apply_single never reads it.
Thus stock SGD has one weight-sized state allocation in these measurements.

Add a subclass which omits that allocation only when momentum is zero. Verify
three updates against stock SGD, then run the same five dense/independent/stacked
and all/one activity cases through the unchanged frozen benchmark utility.
This control distinguishes a mathematical need from an implementation choice.
It is not a convergence experiment or a claim about all SGD implementations.
Freeze the amendment and adapter before running. Keep initial source/results.
