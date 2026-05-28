from __future__ import annotations

from collections import namedtuple
from math import ceil, log2
from typing import Callable

import numpy as np
from tinygrad import Tensor, dtypes, nn


Return = namedtuple("Return", ["quantized", "indices", "entropy_aux_loss"])
LossBreakdown = namedtuple("LossBreakdown", ["per_sample_entropy", "batch_entropy", "commitment"])


# Small helpers keep the LFQ class readable without pulling in einops-style
# dependencies for a handful of reshape/permute cases.
def exists(x):
    return x is not None


# Matches the PyTorch helper: return the first non-None value, evaluating lazy
# defaults only when needed.
def default(*xs):
    for x in xs:
        if exists(x):
            return x() if callable(x) else x
    return None


# Entropy over the last axis.  Input probabilities (..., K) produce (...).
def entropy(prob, eps=1e-5):
    return -(prob * prob.maximum(eps).log()).sum(axis=-1)


# Constant LFQ bit weights and {-scale,+scale} codes are easiest to seed from
# numpy; all forward math after construction stays in tinygrad Tensor ops.
def build_codebook(codebook_size, codebook_dim, scale):
    mask = (2 ** np.arange(codebook_dim - 1, -1, -1)).astype(np.int32)
    codes = np.arange(codebook_size, dtype=np.int32)
    bits = ((codes[:, None] & mask[None, :]) != 0).astype(np.float32)
    return mask, bits * (2.0 * scale) - scale


class LFQ:
    # Builds the tinygrad LFQ module with the key PyTorch options: projections,
    # codebook packing, entropy loss, commitment loss, and spatial handling.
    def __init__(
        self,
        *,
        dim=None,  # input feature dimension; defaults to num_codebooks * log2(codebook_size)
        codebook_size=None,  # number of discrete codes, must be a power of two
        entropy_loss_weight=0.1,  # multiplier for the LFQ entropy/diversity auxiliary loss
        commitment_loss_weight=0.0,  # multiplier for MSE between encoder outputs and hard codes
        diversity_gamma=1.0,  # weight for rewarding high batch-level codebook usage
        straight_through_activation: Callable[[Tensor], Tensor] | None = None,  # optional activation before straight-through
        num_codebooks=1,  # number of independent LFQ codebooks packed into the feature dim
        keep_num_codebooks_dim=None,  # keep indices as (..., num_codebooks) instead of squeezing single codebook
        codebook_scale=1.0,  # absolute value of each binary code component
        frac_per_sample_entropy=1.0,  # fraction of tokens used for per-sample entropy estimate
        has_projections=None,  # use Linear projections when dim differs from packed codebook dims
        projection_has_bias=True,  # include bias terms in projection layers
        soft_clamp_input_value=None,  # tanh-clamp projected inputs to +/- this value
        channel_first=None,  # treat spatial inputs as B,D,*S when true; auto-detect by default
        spherical=False,  # normalize code vectors and inputs onto a sphere
        **unsupported,  # accepted PyTorch LFQ options that this concise port may ignore or reject
    ):
        if unsupported:
            ignored = {"force_quantization_f32"}
            enabled = [k for k, v in unsupported.items() if k not in ignored and v]
            if enabled:
                raise NotImplementedError(f"unsupported LFQ options: {enabled}")

        if not exists(dim) and not exists(codebook_size):
            raise ValueError("either dim or codebook_size must be specified for LFQ")
        if exists(codebook_size) and (codebook_size <= 0 or codebook_size & (codebook_size - 1)): # last check checks for a power of 2, bit hack
            suggested = 2 ** ceil(log2(codebook_size))
            raise ValueError(f"codebook_size must be a power of 2, suggested {suggested}")

        codebook_size = default(codebook_size, lambda: 2**dim)
        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        has_projections = default(has_projections, dim != codebook_dims)

        self.dim, self.codebook_dim, self.codebook_size = dim, codebook_dim, codebook_size
        self.num_codebooks, self.codebook_scale = num_codebooks, float(codebook_scale)
        self.keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        if num_codebooks > 1 and not self.keep_num_codebooks_dim:
            raise ValueError("num_codebooks > 1 requires keep_num_codebooks_dim=True")

        self.channel_first = channel_first
        self.spherical = spherical
        self.activation = straight_through_activation or (lambda x: x)
        self.frac_per_sample_entropy = float(frac_per_sample_entropy)
        if not 0.0 < self.frac_per_sample_entropy <= 1.0:
            raise ValueError("frac_per_sample_entropy must be in (0, 1]")

        self.diversity_gamma = float(diversity_gamma)
        self.entropy_loss_weight = float(entropy_loss_weight)
        self.commitment_loss_weight = float(commitment_loss_weight)
        self.soft_clamp_input_value = soft_clamp_input_value
        if exists(soft_clamp_input_value) and soft_clamp_input_value < codebook_scale:
            raise ValueError("soft_clamp_input_value must be >= codebook_scale")

        self.has_projections = bool(has_projections)
        self.project_in = nn.Linear(dim, codebook_dims, bias=projection_has_bias) if self.has_projections else None
        self.project_out = nn.Linear(codebook_dims, dim, bias=projection_has_bias) if self.has_projections else None

        mask, codebook = build_codebook(codebook_size, codebook_dim, self.codebook_scale)
        self.mask = Tensor(mask, dtype=dtypes.int32).is_param_(False)
        self.codebook = Tensor(codebook, dtype=dtypes.float32).is_param_(False)

    # Binary LFQ maps 0/1 bits to actual code values in {-scale,+scale}.
    def bits_to_codes(self, bits):
        return bits.cast(dtypes.float32) * (2.0 * self.codebook_scale) - self.codebook_scale

    # Optional BSQ-style spherical normalization keeps code vectors on a sphere.
    # Shape is unchanged: (..., D) -> (..., D).
    def maybe_l2norm(self, x):
        if not self.spherical:
            return x
        denom = (x * x).sum(axis=-1, keepdim=True).maximum(1e-12).sqrt()
        return x / denom * self.codebook_scale

    # Tinygrad has modules as plain callables, so projections are explicit.
    # Input/output shape: (..., dim) <-> (..., num_codebooks * codebook_dim).
    def _project_in(self, x):
        return self.project_in(x) if self.project_in is not None else x

    # Projects quantized codebook dims back to the user feature dimension.
    # Input/output shape: (..., num_codebooks * codebook_dim) -> (..., dim).
    def _project_out(self, x):
        return self.project_out(x) if self.project_out is not None else x

    # Standardizes sequence/image/video inputs to B,N,D for quantization.
    # B,D,*S channel-first becomes B,N,D and records S for later restoration.
    def _flatten_input(self, x):
        is_spatial = x.ndim >= 4
        should_transpose = default(self.channel_first, is_spatial)
        if not is_spatial:
            return x, None

        if should_transpose:
            order = (0, *range(2, x.ndim), 1)
            x = x.permute(order)
        spatial_shape = x.shape[1:-1]
        return x.reshape(x.shape[0], int(np.prod(spatial_shape)), x.shape[-1]), (spatial_shape, should_transpose)

    # Restores B,N,D outputs and B,N,C indices back to spatial layout.
    # Example: B,N,D with S=(H,W) returns B,D,H,W when channel_first is true.
    def _restore_output(self, x, indices, spec):
        if spec is None:
            return x, indices

        spatial_shape, should_transpose = spec
        x = x.reshape(x.shape[0], *spatial_shape, x.shape[-1])
        indices = indices.reshape(indices.shape[0], *spatial_shape, *indices.shape[2:])
        if should_transpose:
            x = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1))
        return x, indices

    # Converts mask shaped B or B,N into float token weights for loss terms.
    # Output shape is B,N,1,1 so it broadcasts over codebooks and bits.
    def _mask_weights(self, mask, x):
        if mask is None:
            return None
        mask = mask.cast(dtypes.float32)
        if mask.ndim == 1:
            mask = mask.reshape(mask.shape[0], 1).expand((x.shape[0], x.shape[1]))
        if mask.shape != x.shape[:2]:
            raise ValueError(f"mask must have shape B or B,N; got {mask.shape}, expected {x.shape[:2]}")
        return mask.reshape(mask.shape[0], mask.shape[1], 1, 1)

    # Computes LFQ entropy regularization over soft assignments to all codes.
    # Input x is B,N,C,D; probabilities are (B*N),C,K.
    def _entropy_loss(self, x, inv_temperature, weights):
        seq = x.reshape(x.shape[0] * x.shape[1], self.num_codebooks, self.codebook_dim)
        flat_weights = None if weights is None else weights.reshape(x.shape[0] * x.shape[1], 1, 1)

        if self.frac_per_sample_entropy < 1.0:
            n = max(1, int(seq.shape[0] * self.frac_per_sample_entropy))
            seq = seq[:n]
            flat_weights = None if flat_weights is None else flat_weights[:n]

        codebook = self.maybe_l2norm(self.codebook)
        probs = (seq.matmul(codebook.T) * (2.0 * inv_temperature)).softmax(axis=-1)

        if flat_weights is None:
            per_sample_entropy = entropy(probs).mean()
            avg_prob = probs.mean(axis=0)
        else:
            denom = flat_weights.sum()
            per_sample_entropy = (entropy(probs) * flat_weights.reshape(flat_weights.shape[0], 1)).sum() / (denom * self.num_codebooks)
            avg_prob = (probs * flat_weights).sum(axis=0) / denom

        batch_entropy = entropy(avg_prob).mean()
        return per_sample_entropy - self.diversity_gamma * batch_entropy, per_sample_entropy, batch_entropy

    # Commitment keeps encoder outputs near their hard codes.
    # Both inputs are B,N,C,D; optional weights are B,N,1,1.
    def _commitment_loss(self, original, quantized, weights):
        loss = (original - quantized.detach()).square()
        if weights is None:
            return loss.mean()
        return (loss * weights).sum() / (weights.sum() * self.num_codebooks * self.codebook_dim)

    # Maps packed integer code IDs back to code vectors, matching PyTorch LFQ.
    # Indices ...[,C] become codes ...,(C*D), then optionally project out.
    def indices_to_codes(self, indices, project_out=True):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)
        if not self.keep_num_codebooks_dim:
            indices = indices.unsqueeze(-1)

        bits = ((indices.cast(dtypes.int32)[..., None] & self.mask) != 0).cast(dtypes.float32)
        codes = self.maybe_l2norm(self.bits_to_codes(bits))
        codes = codes.reshape(*codes.shape[:-2], self.num_codebooks * self.codebook_dim)
        codes = self._project_out(codes) if project_out else codes

        if should_transpose:
            codes = codes.permute(0, codes.ndim - 1, *range(1, codes.ndim - 1))
        return codes

    # Forward quantizes B,N,D or B,D,*S inputs and returns PyTorch-like
    # (quantized, indices, aux_loss); indices are B,N[,C] or B,*S[,C].
    def __call__(self, x, inv_temperature=100.0, return_loss_breakdown=False, mask=None):
        x, spatial_spec = self._flatten_input(x)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected dimension {self.dim}, received {x.shape[-1]}")

        x = self._project_in(x)
        if exists(self.soft_clamp_input_value):
            x = (x / self.soft_clamp_input_value).tanh() * self.soft_clamp_input_value

        x = x.reshape(x.shape[0], x.shape[1], self.num_codebooks, self.codebook_dim)
        original = self.maybe_l2norm(x)
        weights = self._mask_weights(mask, original)

        bits = (original > 0).cast(dtypes.float32)
        quantized = self.maybe_l2norm(self.bits_to_codes(bits))
        indices = (bits.cast(dtypes.int32) * self.mask.reshape(1, 1, 1, self.codebook_dim)).sum(axis=-1).cast(dtypes.int32)

        if Tensor.training:
            activated = self.activation(original)
            x = activated + (quantized - activated).detach()
            entropy_aux, per_sample_entropy, batch_entropy = self._entropy_loss(original, inv_temperature, weights)
            commit_loss = self._commitment_loss(original, quantized, weights) if self.commitment_loss_weight > 0.0 else original.sum() * 0.0
        else:
            x = quantized
            entropy_aux = per_sample_entropy = batch_entropy = commit_loss = original.sum() * 0.0

        x = x.reshape(x.shape[0], x.shape[1], self.num_codebooks * self.codebook_dim)
        x = self._project_out(x)
        x, indices = self._restore_output(x, indices, spatial_spec)
        if not self.keep_num_codebooks_dim:
            indices = indices.squeeze(-1)

        aux_loss = entropy_aux * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight
        ret = Return(x, indices, aux_loss)
        if not return_loss_breakdown:
            return ret
        return ret, LossBreakdown(per_sample_entropy, batch_entropy, commit_loss)
