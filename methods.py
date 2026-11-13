"""Independent component switches for baselines and the two-factor ablation."""

METHODS = {
    "fedavg": {"prior": False, "curriculum": False, "proximal": False},
    "fedprox": {"prior": False, "curriculum": False, "proximal": True},
    "fedcma": {"prior": True, "curriculum": True, "proximal": False},
    "fedcma_wo_prior": {"prior": False, "curriculum": True, "proximal": False},
    "fedcma_wo_cl": {"prior": True, "curriculum": False, "proximal": False},
    "fedcma_wo_both": {"prior": False, "curriculum": False, "proximal": False},
}


def method_components(method):
    return METHODS[method].copy()
