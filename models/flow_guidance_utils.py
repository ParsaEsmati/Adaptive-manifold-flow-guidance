"""
Guidance rules for flow-matching samplers (Stable Diffusion 3 / FLUX).

All functions take the conditional and unconditional velocity predictions of one
denoising step and return the velocity the Euler update should use. The proposed
method is `apply_cfg_manifold_guidance_v`; the others are the baselines compared
against in the paper (APG, Rect-CFG++, limited-interval guidance) and an
x_0-space variant of the method.
"""
from typing import Optional

import torch
import torch.nn.functional
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0
    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


class StepCounter:
    def __init__(self, state: bool = True):
        # True -> the first step is the guided one
        self.state = state
    def update(self):
        self.state = not self.state


def current_step_index(scheduler: FlowMatchEulerDiscreteScheduler) -> int:
    # scheduler.step_index is only initialised inside scheduler.step(), so it is
    # still None on the first denoising iteration. Resolve it here so every
    # guidance method applies its own rule from step 0 instead of silently
    # falling back to plain CFG there.
    if scheduler.step_index is not None:
        return scheduler.step_index
    return scheduler.begin_index or 0


def project(
    v0: torch.Tensor, # [B, C, H, W]
    v1: torch.Tensor, # [B, C, H, W]
    ):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def vector_decomposition(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Decompose `velocity` into components along and orthogonal to `latents`,
    *per spatial/temporal position*.

    Automatically handles both 3D and 4D tensors.

    Args:
        latents: Direction tensor, same shape as velocity
                 [B, C, S] or [B, C, H, W]
        velocity: Vector to decompose, same shape as latents

    Returns:
        v_along:   component of `velocity` along `latents` at each position
        v_orth:    component of `velocity` orthogonal to `latents` at each position

        (returned in this order to stay compatible with existing call sites)
    """
    if latents.shape != velocity.shape:
        raise ValueError(
            f"latents and velocity must have the same shape, "
            f"got {latents.shape} vs {velocity.shape}"
        )

    is_4d = (latents.ndim == 4)

    if is_4d:
        # 4D: [B, C, H, W] -> [B, C, S] with S = H*W
        B, C, H, W = latents.shape
        latents_flat  = latents.view(B, C, -1)   # [B, C, S]
        velocity_flat = velocity.view(B, C, -1)  # [B, C, S]
    else:
        # 3D: [B, C, S] - already flat
        latents_flat  = latents
        velocity_flat = velocity

    latent_norm = latents_flat.norm(dim=1, keepdim=True)
    latents_hat = latents_flat / (latent_norm + eps)

    dot = (velocity_flat * latents_hat).sum(dim=1, keepdim=True)
    v_along_flat = dot * latents_hat
    v_orth_flat = velocity_flat - v_along_flat

    # Reshape back to original shape
    if is_4d:
        v_along = v_along_flat.view(B, C, H, W)
        v_orth  = v_orth_flat.view(B, C, H, W)
    else:
        v_along = v_along_flat
        v_orth  = v_orth_flat
    return v_along, v_orth


def apply_cfg_manifold_guidance_v(
    x: torch.Tensor,
    t: torch.Tensor,
    velocity_cond: torch.Tensor,
    velocity_uncond: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    guidance_scale_tangent: float = 0.1,
    guidance_scale: float = 7.0,
    eps: float = 1e-8,
    scale_damp_mode: str = "power",
    scale_min_value: float = 4.0,
    gamma: float = 1.0,
    tangent_scale_mode: str = "ours",
    basis: str = "cond",
) -> torch.Tensor:
    """
    Manifold-aware classifier-free guidance in velocity space (the proposed method).

    The CFG residual g = v_cond - v_uncond is decomposed, per latent position, into
    its component c along the predicted-noise direction (the score-parallel part,
    g_par in the paper) and the orthogonal remainder d (g_perp), and the two are
    re-weighted:

        v = v_uncond + beta * c + omega(t) * d

    beta damps the score-parallel part; omega(t) is the guidance schedule on the
    orthogonal part. With beta = omega and a constant omega this is plain CFG.

    basis selects whose predicted noise the residual is decomposed against:
    "cond" (eps from velocity_cond, the default) or "uncond" (eps from velocity_uncond).

    tangent_scale_mode selects how guidance_scale_tangent (beta) is interpreted:
    - "ours": beta is the raw coefficient of c (the paper's beta)
    - "apg":  the coefficient becomes 1 + (omega - 1) * beta, i.e. the value means
              the same thing as APG's eta, so numbers transfer between the methods.

    Guidance-scale schedule omega(t) (scale_damp_mode), with s = t / 1000 in [0, 1]:
    - "power":  omega = max(scale_min_value, guidance_scale * s ** gamma)
    - "cutoff": omega = guidance_scale while (1 - s) > gamma, then scale_min_value
    - anything else: constant omega = guidance_scale
    """
    alpha = 1 - (t / 1000.0)
    g_raw = velocity_cond - velocity_uncond

    if scale_damp_mode.lower() == "power":
        effective_guidance_scale = torch.clamp(
            guidance_scale * ((t / 1000.0) ** gamma), min=scale_min_value
        )
    elif scale_damp_mode.lower() == "cutoff":
        effective_guidance_scale = torch.tensor(
            max(scale_min_value, guidance_scale) if alpha > gamma else scale_min_value
        )
    else:
        effective_guidance_scale = torch.tensor(guidance_scale)

    v_cfg = velocity_uncond + effective_guidance_scale * g_raw

    step_index = current_step_index(scheduler)
    sigma_t = scheduler.sigmas[step_index]
    sigma_final = scheduler.sigmas[-1]
    delta_sigma = sigma_t - sigma_final
    if abs(float(delta_sigma)) < eps:
        return v_cfg

    # Decomposition basis: the predicted NOISE direction. With sigma = t/1000 and
    # the flow-matching interpolant x = (1 - sigma) x0 + sigma eps, v = eps - x0, so
    #     n = x + (1 - sigma) * velocity_cond = eps_cond   (exact)
    # This is alpha * (x / alpha + velocity_cond); projection is scale-invariant, so
    # the direction is the same, but this form stays finite at the first step, where
    # alpha = 0. Contrast APG, which decomposes against x0_cond = x - sigma * v: the
    # two bases are the opposite ends of the same parameterisation.
    n = x + alpha * (velocity_uncond if basis == "uncond" else velocity_cond)
    c, d = vector_decomposition(n, g_raw)

    if tangent_scale_mode == "apg":
        tangent_coeff = 1.0 + (effective_guidance_scale - 1.0) * guidance_scale_tangent
    else:
        tangent_coeff = guidance_scale_tangent

    dv_new = tangent_coeff * c + effective_guidance_scale * d
    return velocity_uncond + dv_new


def apply_apg_guidance(
    x: torch.Tensor,
    t: torch.Tensor,
    velocity_cond: torch.Tensor,
    velocity_uncond: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    guidance_scale_tangent: float = 0.1,
    guidance_scale: float = 7.0,
    idx_eval: int = -1,
    momentum_buffer: Optional[MomentumBuffer] = None,
    norm_threshold: Optional[float] = 20.0,
    projection: str = "per-pixel",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Adaptive Projected Guidance (APG, Sadat et al. 2024) in x_0 space.

    Maps velocities to x_0 via one Euler step, decomposes the x_0 residual against
    x0_cond into tangent and normal components, reweights them, and maps back.

    Both APG ingredients are independently switchable:
    - norm_threshold=None (or <= 0) skips the norm clipping of dx0
    - momentum_buffer=None skips the momentum accumulation
    Passing None for both reduces this to the plain tangent/normal reweighting.

    projection selects how dx0 is split against x0_cond:
    - 'per-pixel': at every spatial position, across channels (vector_decomposition)
    - 'global':    one direction per sample over (C, H, W), i.e. the APG paper's
                   project()
    """
    g_raw = guidance_scale * (velocity_cond - velocity_uncond)
    v_cfg = velocity_uncond + g_raw

    step_index = current_step_index(scheduler)

    sigma_t = scheduler.sigmas[step_index]
    sigma_final = scheduler.sigmas[idx_eval]
    delta_sigma = sigma_final - sigma_t

    if abs(float(delta_sigma)) < eps:
        return v_cfg

    x0_uncond = x + delta_sigma * velocity_uncond
    x0_cond = x + delta_sigma * velocity_cond
    x0_cfg = x + delta_sigma * v_cfg
    dx0 = x0_cond - x0_uncond

    if momentum_buffer is not None:
        momentum_buffer.update(dx0)
        dx0 = momentum_buffer.running_average

    if norm_threshold is not None and norm_threshold > 0:
        ones = torch.ones_like(dx0)
        dx0_norm = dx0.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / dx0_norm)
        dx0 = dx0 * scale_factor

    n = x0_cond

    if projection == "global":
        dx0_tangent, dx0_normal = project(dx0, n)
    else:
        dx0_tangent, dx0_normal = vector_decomposition(n, dx0)
    dx0_new = guidance_scale_tangent * dx0_tangent + dx0_normal

    x0_new = x0_cond + (guidance_scale - 1) * dx0_new
    v_new = (x0_new - x) / delta_sigma

    return v_new


def apply_interval_guidance(
    velocity_cond: torch.Tensor,
    velocity_uncond: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    guidance_scale: float = 7.0,
    sigma_lo: float = 0.2188,
    sigma_hi: float = 0.8442,
) -> torch.Tensor:
    """
    Guidance in a limited interval (Kynkaanniemi et al., NeurIPS 2024, Eq. 6):

        w(sigma) = w   if sigma in (sigma_lo, sigma_hi]
                   1   otherwise

    i.e. plain CFG at a CONSTANT scale inside the interval and the purely
    conditional prediction outside it (no guidance at high and at low noise).

    sigma_lo / sigma_hi are in this scheduler's units (sigma = t/1000 in [0, 1]).
    The paper works with the EDM noise level sigma_edm (x = x0 + sigma_edm * eps);
    for the rectified-flow interpolant x = (1 - sigma) * x0 + sigma * eps the same
    signal-to-noise ratio is reached at sigma = sigma_edm / (1 + sigma_edm). The
    defaults are the paper's Stable Diffusion XL interval (0.28, 5.42] mapped this
    way; the paper gives no interval for SD3-class models.
    """
    sigma_t = float(scheduler.sigmas[current_step_index(scheduler)])
    if sigma_lo < sigma_t <= sigma_hi:
        return velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
    return velocity_cond


def apply_rectcfgpp_guidance(
    x: torch.Tensor,
    t: torch.Tensor,
    velocity_cond: torch.Tensor,
    velocity_uncond: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_fn,
    guidance_scale: float = 7.0,
    sigma_noise: float = 0.005,
) -> torch.Tensor:
    """
    Rectified-CFG++ (Saini, Gupta, Bovik, NeurIPS 2025), as in the authors' released SD3
    pipeline (github.com/shreshthsaini/Rectified-CFGpp, rect-cfg-SD3-pipeline/pipeline.py):

        dt     = sigma_{i+1} - sigma_i
        x_pred = x + dt * v_c(x, t)                    predictor: conditional Euler step
        x_pred = x_pred + sigma_noise * N(0, I)        (their default 0.005; off once t <= 0.1)
        g      = v_c(x_pred, t') - v_u(x_pred, t')     corrector: guidance difference at x_pred
        v_hat  = v_c(x, t) + w * g                     w = their `true_cfg`
        x_next = scheduler.step(v_hat) = x + dt * v_hat   (done by the pipeline, as in theirs)

    `guidance_scale` is w itself, applied exactly as their `true_cfg`: v_c + w * g. (Plain CFG at
    scale s is v_u + s * g = v_c + (s - 1) * g, so Rect-CFG++ at w kicks like CFG at s = w + 1.)
    `t'` reproduces their corrector timestep `t + dt` verbatim: t is on the 0..1000 scale and dt
    is a sigma difference, so numerically t' = t. Two model evaluations per step;
    `model_fn(x, t) -> (v_uncond, v_cond)` performs the second one.

    The paper's Algorithm 1 is printed with a half-step predictor (x + dt/2 * v_c), the
    corrector at t - dt/2 and a weight schedule alpha(t); the released code, followed here,
    uses the full step, the same timestep and a constant weight.
    """
    step_index = current_step_index(scheduler)
    sigmas = scheduler.sigmas
    sigma_t = sigmas[step_index]
    sigma_next = sigmas[step_index + 1] if step_index + 1 < len(sigmas) else torch.zeros_like(sigma_t)
    dt = sigma_next - sigma_t

    x_pred = x + dt * velocity_cond
    if sigma_noise > 0 and float(t) > 0.1:
        x_pred = x_pred + sigma_noise * torch.randn_like(x_pred)
    t_corr = (t + dt).to(x.dtype)

    v_uncond_pred, v_cond_pred = model_fn(x_pred, t_corr)
    return velocity_cond + guidance_scale * (v_cond_pred - v_uncond_pred)


def apply_rectcfgpp_alternating(
    x: torch.Tensor,
    t: torch.Tensor,
    velocity_cond: torch.Tensor,
    velocity_uncond: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    guidance_scale_tangent: float = 0.1,
    guidance_scale: float = 7.0,
    idx_eval: int = -1,
    step_counter: StepCounter = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    The repo's ORIGINAL "rectcfgpp": alternate a full CFG Euler step and a conditional-only
    Euler step. This is NOT Rectified-CFG++ (its conditional step consumes a schedule
    interval, so only every other interval is guided); kept as `rectcfgpp_alt` so the runs
    made with it stay reproducible. See apply_rectcfgpp_guidance for the faithful rule.
    """
    g_raw = guidance_scale * (velocity_cond - velocity_uncond)
    v_cfg = velocity_uncond + g_raw
    if step_counter.state:
        v_new = v_cfg
    else:
        v_new = velocity_cond
    step_counter.update()
    return v_new


def apply_cfg_manifold_guidance_x0(
    x: torch.Tensor,
    t: torch.Tensor,
    velocity_cond: torch.Tensor,
    velocity_uncond: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    guidance_scale_tangent: float = 0.1,
    guidance_scale: float = 7.0,
    isomap: str = "iso_x0",
    idx_eval: int = -1,
    x0_basis: str = "x0_cond",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    The x_0-space variant of the proposed method.

    Maps velocities to x_0 space via one Euler step, decomposes the x_0 residual
    into tangent and normal components, reweights them, and maps back to velocity.

    The decomposition basis 'n' can be chosen via isomap parameter:
    - "iso_xt": xt-based (n = a_t * x_t - v_cond), uses idx_eval for sigma selection
    - "iso_x0": x0-based, uses x0_basis to select which x0 variant (x0_cond, x0_uncond, or x0_cfg)
    """
    g_raw = guidance_scale * (velocity_cond - velocity_uncond)
    v_cfg = velocity_uncond + g_raw

    step_index = current_step_index(scheduler)

    sigma_t = scheduler.sigmas[step_index]
    sigma_final = scheduler.sigmas[idx_eval]
    delta_sigma = sigma_final - sigma_t

    if abs(float(delta_sigma)) < eps:
        return v_cfg

    x0_uncond = x + delta_sigma * velocity_uncond
    x0_cond = x + delta_sigma * velocity_cond
    x0_cfg = x + delta_sigma * v_cfg
    dx0 = x0_cfg - x0_cond

    alpha = t / 1000.0
    alpha_dot = 1.0
    a = alpha_dot / (alpha + eps)

    if isomap == "iso_xt":
        n = a * x - velocity_cond
    else:
        # iso_x0 mode: select basis based on x0_basis parameter
        if x0_basis == "x0_uncond":
            n = x0_uncond
        elif x0_basis == "x0_cfg":
            n = x0_cfg
        else:  # x0_cond
            n = x0_cond

    dx0_tangent, dx0_normal = vector_decomposition(n, dx0)
    dx0_new = guidance_scale_tangent * dx0_tangent + dx0_normal

    x0_new = x0_cond + dx0_new
    v_new = (x0_new - x) / delta_sigma

    return v_new
