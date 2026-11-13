"""Curriculum scheduling and deterministic sampling, including the legacy mode."""
import torch


def curriculum_plan(cfg, round_index, enabled, max_missing_fraction):
    """Use only the training masks to determine the reachable score bound."""
    version = getattr(cfg, "curriculum_schedule", "legacy")
    if not enabled:
        return {"phase": "disabled", "active": False, "rho": 1., "tolerance": 0.}
    if version == "legacy":
        fraction = round_index / max(cfg.rounds - 1, 1)
        return {"phase": "curriculum", "active": True,
                "rho": cfg.rho0 + (1 - cfg.rho0) * fraction, "tolerance": 0.}
    if version == "balanced_weight":
        if not 0 <= cfg.warmup_rounds < cfg.rounds:
            raise ValueError("Weight curriculum requires 0 <= warmup_rounds < rounds")
        if not 0 <= cfg.curriculum_weight_max < 1:
            raise ValueError("curriculum_weight_max must be in [0,1)")
        progress = max(0, round_index - cfg.warmup_rounds + 1) / (cfg.rounds - cfg.warmup_rounds)
        strength = cfg.curriculum_weight_max * progress
        return {"phase": "warmup" if round_index < cfg.warmup_rounds else "online_weighting",
                "active": strength > 0, "rho": 1., "tolerance": 0., "weight_strength": strength,
                "score_source": "current_training_batch_bce_rank_within_label_mask_group",
                "coverage": "every_sample_once_per_epoch", "order": "unchanged_random_order"}
    if version == "stratified_order":
        warmup, recovery = cfg.warmup_rounds, cfg.full_data_rounds
        active_rounds = cfg.rounds - warmup - recovery
        if warmup < 0 or recovery < 1 or active_rounds < 1:
            raise ValueError("Ordering curriculum needs warmup >= 0, full_data_rounds >= 1, and an active round")
        if not 0 <= cfg.curriculum_strength <= 1:
            raise ValueError("curriculum_strength must be in [0,1]")
        upper = cfg.xi * max_missing_fraction + (1 - cfg.xi)
        strength = 0.
        phase = "warmup" if round_index < warmup else "full_data"
        if warmup <= round_index < cfg.rounds - recovery:
            strength = cfg.curriculum_strength * (1 - (round_index - warmup) / active_rounds)
            phase = "ordered_curriculum" if strength > 0 else "full_data"
        return {"phase": phase, "active": strength > 0, "rho": upper,
                "score_upper_bound": upper, "tolerance": 1e-7, "order_strength": strength,
                "score_normalization": "client_max", "coverage": "every_sample_once_per_epoch"}
    if version != "warmup_bounded":
        raise ValueError(f"Unknown curriculum schedule: {version}")
    warmup = cfg.warmup_rounds
    if not 0 <= warmup < cfg.rounds:
        raise ValueError("warmup_rounds must be nonnegative and smaller than rounds")
    upper = cfg.xi * max_missing_fraction + (1 - cfg.xi)
    if not 0 <= cfg.rho0 <= upper:
        raise ValueError(f"rho0 must be in [0, training score upper bound {upper}]")
    plan = {"rho": upper, "score_upper_bound": upper, "tolerance": 1e-7}
    if round_index < warmup:
        return {**plan, "phase": "warmup", "active": False}
    active_rounds = cfg.rounds - warmup
    fraction = (round_index - warmup) / (active_rounds - 1) if active_rounds > 1 else 1.
    return {**plan, "phase": "curriculum", "active": True,
            "rho": cfg.rho0 + (upper - cfg.rho0) * fraction}


def training_order(selected, target_count, generator):
    """Shuffle without replacement per cycle until the exact budget is filled.

    A full eligible pool follows exactly the original one-permutation path.
    A restricted pool repeats each eligible sample floor/ceil(n / pool_size)
    times per epoch. No excluded sample is silently reintroduced.
    """
    if target_count <= 0 or len(selected) == 0:
        raise ValueError("A positive budget requires a nonempty selected pool")
    cycles, remainder = divmod(target_count, len(selected))
    parts = [selected[torch.randperm(len(selected), generator=generator, device=selected.device)]
             for _ in range(cycles)]
    if remainder:
        order = torch.randperm(len(selected), generator=generator, device=selected.device)
        parts.append(selected[order[:remainder]])
    return torch.cat(parts)


def joint_groups(labels, masks):
    """Training-label x observed-modality pattern. Never uses evaluation labels."""
    bits = 2 ** torch.arange(masks.shape[1], device=masks.device)
    return labels.to(torch.int64) * (2 ** masks.shape[1]) + (masks.to(torch.int64) * bits).sum(1)


def stratified_curriculum_order(base_order, local_indices, local_scores, labels, masks, strength):
    """Interpolate random rank and difficulty rank inside label/mask strata.

    The label/mask of each position matches the ordinary random permutation,
    so even each mini-batch's stratum counts remain identical to the control.
    Every training sample appears exactly once; difficulty only changes order.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0,1]")
    if strength == 0:
        return base_order
    if len(base_order) != len(local_indices) or len(local_scores) != len(local_indices):
        raise ValueError("Ordering curriculum requires the full local pool")
    score_by_index = torch.empty_like(labels)
    score_by_index[local_indices] = local_scores
    groups = joint_groups(labels, masks)[base_order]
    ordered = base_order.clone()
    for group in torch.unique(groups):
        slots = torch.where(groups == group)[0]
        members = base_order[slots]
        random_rank = torch.arange(len(slots), dtype=local_scores.dtype, device=slots.device)
        difficulty_rank = torch.empty_like(random_rank)
        difficulty_rank[torch.argsort(score_by_index[members], stable=True)] = random_rank
        priority = (1 - strength) * random_rank + strength * difficulty_rank
        ordered[slots] = members[torch.argsort(priority, stable=True)]
    return ordered


@torch.no_grad()
def balanced_hardness_weights(losses, labels, masks, strength):
    """Bounded online difficulty weights with exact group mass in real arithmetic.

    For each label x mask group, use average ranks for ties, centered in [-1,1].
    w = 1 + strength * rank. Every group's mean weight is one, singleton/tied
    groups retain unit weight, and no sample gets zero weight for strength < 1.
    The ordering and number of training samples are unchanged.
    """
    if not 0 <= strength < 1:
        raise ValueError("strength must be in [0,1)")
    if strength == 0:
        return torch.ones_like(losses)
    groups = joint_groups(labels, masks)
    same = groups[:, None] == groups[None, :]
    less = ((losses[:, None] > losses[None, :]) & same).sum(1).to(losses.dtype)
    ties = ((losses[:, None] == losses[None, :]) & same).sum(1).to(losses.dtype)
    count = same.sum(1).to(losses.dtype)
    average_rank = less + (ties - 1) / 2
    centered = torch.where(count > 1, 2 * average_rank / (count - 1).clamp_min(1) - 1,
                           torch.zeros_like(losses))
    return 1 + strength * centered
