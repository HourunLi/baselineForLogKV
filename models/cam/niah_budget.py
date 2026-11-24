"""CaM fixed storage and the six nominal RULER operating points."""
from fractions import Fraction

TARGET = {32768: "4.08", 65536: "7.11", 131072: "12.37", 262144: "21.57",
          524288: "38.38", 1048576: "69.43"}


def cache_budget(mode, nominal, window, layers, groups, head_dim, element_bytes, reference_bytes):
    pair = layers * groups * 2 * head_dim * element_bytes
    entry = pair + layers * groups * 4
    target = Fraction(TARGET[nominal])
    if mode == "logkv":
        capacity = reference_bytes // entry
    elif mode == "target-bytes":
        capacity = int(nominal * pair / target // entry)
    elif mode == "target-slots":
        capacity = int(nominal // target)
    else:
        raise ValueError(f"Unknown CaM budget mode {mode!r}")
    capacity = min(capacity, window)
    if capacity < 48:
        raise ValueError("Budget cannot hold the 32 sink and 16 merge slots")
    size = capacity * entry
    return dict(method="CaM", allocation="fixed", budget_mode=mode, capacity_tokens=capacity,
                bytes_per_entry=entry, persistent_cache_bytes=size, reference_logkv_bytes=reference_bytes,
                dense_cache_bytes=window * pair, nominal_dense_cache_bytes=nominal * pair,
                allocated_fraction_of_dense=size / (window * pair),
                reference_fraction_of_dense=reference_bytes / (window * pair),
                nominal_allocated_compression=nominal * pair / size,
                nominal_logical_compression=nominal / capacity, nominal_target=float(target))
