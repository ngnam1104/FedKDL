"""physics_models — Mô hình vật lý kênh truyền sóng âm IoUT."""

from physics_models.communication import (
    thorp_absorption, transmission_loss, wenz_noise_level,
    snr_passive, shannon_capacity, min_source_level, is_link_feasible,
)
from physics_models.topology import (
    Topology3D, LinkInfo, build_feasibility_graph,
    nearest_feasible_association, flat_topology_association,
    build_clusters, get_topology_stats,
)
from physics_models.energy import (
    acoustic_power_watts, e_tx, e_rx, e_comp_dynamic, e_comp_full, total_energy_round,
)
from physics_models.latency import comm_delay, comp_delay_simple, round_delay
