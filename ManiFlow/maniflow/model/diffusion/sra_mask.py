"""
Token-group masking utilities for SRA/DTS/AS-style self-distillation training.

Ports the masking/grouping math from SRA/SiT-SRA_DTS_AS's ``loss.py`` (see
``SRA/SiT-SRA_DTS_AS/loss.py``: ``_normalize_mask_ratios``,
``_compute_group_counts``, ``_build_group_ids_from_counts``) so that
ManiFlowTransformerImagePolicy can mix multiple noise levels into a single
action-horizon sequence the same way SiT-SRA_DTS_AS mixes multiple noise
levels into a single image's patch tokens.

The only structural difference from the SiT version is the tensor layout:
SiT groups spatial patch tokens (and stores group ids as (B, T, 1, 1) to
broadcast against (B, T, C, ...) images); here we group temporal
action-horizon tokens and keep group ids as plain (B, T) so callers can
broadcast them against (B, T) scalars or (B, T, D) trajectories themselves
(see ``mix_group_values``).
"""
from typing import List, Sequence, Union

import torch


def normalize_mask_ratios(mask_ratio: Union[float, int, Sequence[float]]) -> List[float]:
    """
    Normalize a mask-ratio spec into a list of per-group fractions that sum to 1.

    Mirrors SiT-SRA_DTS_AS's ``_normalize_mask_ratios``:
      - a single scalar ``r`` in (0, 1) becomes two groups ``[r, 1 - r]``
      - a scalar <= 0 or >= 1 (or an empty/degenerate list) means "no masking",
        i.e. a single group ``[1.0]``
      - a list of ratios is renormalized to sum to 1 (dropping non-positive entries)
    """
    if isinstance(mask_ratio, (int, float)):
        ratios = [float(mask_ratio)]
    else:
        ratios = [float(r) for r in mask_ratio]

    if not ratios:
        raise ValueError("mask_ratio must contain at least one value")
    if any(r < 0 for r in ratios):
        raise ValueError(f"mask_ratio should be non-negative, got {ratios}")

    if len(ratios) == 1:
        ratio = ratios[0]
        if ratio <= 0.0 or ratio >= 1.0:
            return [1.0]
        return [ratio, 1.0 - ratio]

    positive_ratios = [r for r in ratios if r > 0.0]
    if not positive_ratios:
        return [1.0]

    total = sum(positive_ratios)
    return [r / total for r in positive_ratios]


def compute_group_counts(ratios: Sequence[float], seq_len: int, device, batch_size: int) -> torch.Tensor:
    """Per-sample token counts for each group, summing exactly to seq_len."""
    ratios_t = torch.as_tensor(ratios, device=device, dtype=torch.float32).unsqueeze(0)
    ratios_t = ratios_t.expand(batch_size, -1)

    expected = ratios_t * seq_len
    counts = torch.floor(expected).long()
    remainders = seq_len - counts.sum(dim=1)

    for batch_idx, remainder in enumerate(remainders.tolist()):
        if remainder <= 0:
            continue
        fractional = expected[batch_idx] - counts[batch_idx].float()
        topk = torch.topk(fractional, k=remainder).indices
        counts[batch_idx, topk] += 1

    return counts


def build_group_ids_from_counts(counts: torch.Tensor, seq_len: int, device) -> torch.Tensor:
    """
    Randomly assign each of the seq_len positions to a group, subject to the
    per-sample per-group counts. Returns (B, T) long group ids.
    """
    batch_size, num_groups = counts.shape
    perm = torch.rand((batch_size, seq_len), device=device).argsort(dim=1)
    ordered_group_ids = torch.empty((batch_size, seq_len), device=device, dtype=torch.long)
    rank = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    start = torch.zeros((batch_size, 1), device=device, dtype=torch.long)

    for group_idx in range(num_groups):
        end = start + counts[:, group_idx:group_idx + 1]
        in_group = (rank >= start) & (rank < end)
        ordered_group_ids[in_group] = group_idx
        start = end

    group_ids = torch.empty_like(ordered_group_ids)
    group_ids.scatter_(1, perm, ordered_group_ids)
    return group_ids


def build_group_mask(
    mask_ratio: Union[float, int, Sequence[float]],
    batch_size: int,
    seq_len: int,
    device,
    full_sample_prob: float = 0.0,
) -> torch.Tensor:
    """
    Build the (B, T) group-id tensor used to mix noise levels across the
    action horizon. If ``mask_ratio`` normalizes to a single group, every
    position gets group id 0 (i.e. masking is a no-op).

    ``full_sample_prob`` (only valid for exactly two groups, matching
    SiT-SRA_DTS_AS) randomly forces some rows of the batch entirely into
    group 1 (the majority group under the default [0.25, 0.75] split),
    i.e. occasionally trains on an ordinary single-noise-level sample even
    when masking is enabled.
    """
    ratios = normalize_mask_ratios(mask_ratio)
    num_groups = len(ratios)

    if num_groups == 1:
        return torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    if full_sample_prob < 0.0 or full_sample_prob > 1.0:
        raise ValueError(f"full_sample_prob should be in [0, 1], got {full_sample_prob}")
    if full_sample_prob > 0.0 and num_groups != 2:
        raise ValueError("full_sample_prob only supports two-group masks")

    counts = compute_group_counts(ratios, seq_len, device, batch_size=batch_size)
    group_ids = build_group_ids_from_counts(counts, seq_len, device)

    if full_sample_prob > 0.0:
        use_full_sample = torch.rand((batch_size,), device=device) < full_sample_prob
        if use_full_sample.any():
            group_ids[use_full_sample] = 1

    return group_ids


def mix_group_values(group_values: Sequence[torch.Tensor], group_ids: torch.Tensor) -> torch.Tensor:
    """
    Combine per-group tensors into a single tensor by selecting, at each
    (batch, horizon-position), the value from the group that position was
    assigned to.

    ``group_values[k]`` must all share the same shape, which must start with
    (B, T, ...); ``group_ids`` is (B, T) with values in [0, len(group_values)).
    Works for both (B, T) scalar-per-token tensors (e.g. mixed timesteps) and
    (B, T, D) tensors (e.g. mixed trajectories), matching SiT-SRA_DTS_AS's
    ``_mix_group_scalars`` / the token-gather in ``SiT.mix_group_tokens``.
    """
    mixed = group_values[0].clone()
    gid = group_ids
    while gid.dim() < mixed.dim():
        gid = gid.unsqueeze(-1)
    for group_idx in range(1, len(group_values)):
        mixed = torch.where(gid == group_idx, group_values[group_idx], mixed)
    return mixed


def build_attention_separation_mask(group_ids: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Build the boolean attention mask that blocks cross-group self-attention,
    in the layout ``nn.MultiheadAttention`` expects for a per-sample 3D mask:
    (B * num_heads, T, T), True = "not allowed to attend".

    This is the inverse convention of SiT-SRA_DTS_AS's
    ``GroupSeparatedAttention`` (there, True means "allow attend"); here we
    build the "block" mask directly since that's what
    ``nn.MultiheadAttention`` consumes.
    """
    same_group = group_ids[:, :, None] == group_ids[:, None, :]  # (B, T, T), True = same group
    block_mask = ~same_group
    batch_size, seq_len, _ = block_mask.shape
    return block_mask.unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len).reshape(
        batch_size * num_heads, seq_len, seq_len
    )
