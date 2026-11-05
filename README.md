<div align="center">
<h1>Probability-Conserving Flow Guidance</h1>

**Parsa Esmati**\* · **Junha Hyung**\*
<br>
**Amirhossein Dadashzadeh** · **Jaegul Choo** · **Majid Mirmehdi**
<br>
**\*Equal contribution**

**Accepted at NeurIPS 2026**

<a href="https://arxiv.org/abs/2605.20079"><img src="https://img.shields.io/badge/arXiv-2605.20079-b31b1b" alt="arXiv"></a>
</div>

Official implementation of *Probability-Conserving Flow Guidance*. The repository provides
**Adaptive Manifold Guidance (AdaMaG)**, a plug-and-play replacement for classifier-free
guidance in flow-matching samplers, implemented for Stable Diffusion 3 / 3.5 and FLUX.1,
together with the baselines it is compared against in the paper: plain CFG, APG,
Rect-CFG++ and limited-interval guidance. Everything runs through one command-line entry
point on top of the Diffusers pipelines, with no extra inference cost over CFG.

## Method

At every denoising step the CFG residual is split, per latent position, against the
score-proportional direction `n` (the predicted noise of the conditional branch), and the
two parts are weighted separately:

```
g   = v_cond − v_uncond
g∥  = ⟨g, n⟩ / ‖n‖² · n            score-parallel part
g⊥  = g − g∥                       orthogonal part
v   = v_uncond + ω(t) · g⊥ + β · g∥
ω(t) = max(ω_min, ω_ref · σ(t)^γ)  σ(t) = t / 1000, 1 at pure noise
```

`β < 1` damps the score-parallel part, which is what drives samples off the manifold
(over-saturation and hallucinations at high scales); `ω(t)` keeps the orthogonal part at
the reference scale early and decays it towards `ω_min` late in sampling. With `β = ω`
and a constant `ω` the rule is plain CFG. The implementation is
`apply_cfg_manifold_guidance_v` in [`models/flow_guidance_utils.py`](models/flow_guidance_utils.py).

| symbol | flag | SD3 / SD3.5 | FLUX.1 |
|---|---|---|---|
| `ω_ref` | `--guidance-scale` | the CFG scale (paper: 7 and 15) | the true-CFG scale (paper: 2 and 3) |
| `β` | `--tangent-scale` | 0.1 | 0.1 |
| `γ` | `--gamma` (with `--scale-damp-mode power`) | 4 | 6 |
| `ω_min` | `--scale-min-value` | 4 | 1 |

## Installation

```bash
conda env create -f environment.yaml
conda activate flowguide
pip install -e .
```

Tested with `torch 2.7`, `diffusers 0.35`, `transformers 4.57` on a single GPU.

## Model weights

The checkpoints are gated on Hugging Face: request access on the model pages, run
`hf auth login`, then download the Diffusers-format weights and pass the directory with
`--ckpt-dir`.

```bash
hf download stabilityai/stable-diffusion-3-medium-diffusers --local-dir ./sd3_weights
hf download stabilityai/stable-diffusion-3.5-medium          --local-dir ./sd35_weights
hf download black-forest-labs/FLUX.1-dev                     --local-dir ./flux_weights
```

SD3 and SD3.5 both run with `--model sd3`; FLUX.1 with `--model flux`.

## Quick start

A single image with AdaMaG at the paper's SD3 settings:

```bash
python main.py --model sd3 --ckpt-dir ./sd3_weights \
    --prompt "A cat sitting on the left side of a dog looking at each other." \
    --guidance-method v_update --guidance-scale 7 \
    --tangent-scale 0.1 --scale-damp-mode power --gamma 4 --scale-min-value 4 \
    --steps 40 --output cat_dog.png
```

The same for FLUX.1 (true CFG with a negative prompt; the paper's FLUX settings):

```bash
python main.py --model flux --ckpt-dir ./flux_weights \
    --prompt "A cat sitting on the left side of a dog looking at each other." \
    --guidance-method v_update --guidance-scale 3 \
    --tangent-scale 0.1 --scale-damp-mode power --gamma 6 --scale-min-value 1 \
    --steps 40 --output cat_dog_flux.png
```

Plain CFG for comparison is `--guidance-method cfg --guidance-scale 7`. Resolution defaults
to 1024×1024 ([`configs/`](configs/)); override with `--height` / `--width`.

## Batch generation

Pass a prompt file instead of a prompt: a CSV with one prompt per row (optional second
column: output filename), or a COCO captions JSON (`caption` and `image_id` fields).

```bash
python main.py --model sd3 --ckpt-dir ./sd3_weights \
    --prompts-file example_prompts.csv --gen-batch-size 8 \
    --guidance-method v_update --guidance-scale 7 \
    --tangent-scale 0.1 --scale-damp-mode power --gamma 4 --scale-min-value 4 \
    --steps 40 --height 256 --width 256 --output-dir outputs
```

Images are written to `outputs/<run-name>/`, where the run name encodes the guidance
configuration (here `cfg7_v_update_ts0.1_pow4_min4_perimg`), so different settings never
overwrite each other. `--max-prompts N` limits the run; `--skip-existing` resumes an
interrupted one. The paper's quantitative results use 5,000 COCO validation captions at
256×256 with 40 steps.

## Guidance methods

All methods share `--guidance-scale`; the extra flags are listed per method.

| `--guidance-method` | method | extra flags |
|---|---|---|
| `v_update` | **AdaMaG (ours)**, velocity-space rule above | `--tangent-scale` (β), `--scale-damp-mode`, `--gamma`, `--scale-min-value` (ω_min), `--v-basis` |
| `x0_update` | ours, x0-space variant of the decomposition | `--tangent-scale`, `--isomap`, `--x0-basis` |
| `cfg` | plain classifier-free guidance | — |
| `apg` | Adaptive Projected Guidance (Sadat et al., 2024) | `--tangent-scale` (η), `--apg-norm-threshold`, `--apg-momentum`, `--apg-projection` |
| `rectcfgpp` | Rectified-CFG++ (Saini et al., 2025), the authors' released update: conditional predictor step, guidance difference measured at the predicted point; two model evaluations per step, so compare at half the step count for an equal budget | `--guidance-scale` is the authors' `true_cfg` (`v_c + w·g`; CFG at scale `s` corresponds to `w = s − 1`), `--rectcfgpp-sigma-noise` (authors' default 0.005) |
| `rectcfgpp_alt` | alternating sampler, one CFG Euler step then one conditional-only step, one evaluation per step; the evaluation-matched Rect-CFG++ row of the paper's tables | `--rectcfgpp-first` |
| `interval` | limited-interval guidance (Kynkäänniemi et al., 2024): CFG for σ in `(lo, hi]`, conditional velocity elsewhere | `--interval-lo`, `--interval-hi` |

`--guidance-scale 1` disables guidance for every method.

## Reproducibility

Noise is seeded per image: image `k` of a run uses seed `seed · 1000003 + k` with
`--seed` (default 88), so results are independent of `--gen-batch-size` and a
single-prompt run reproduces image 1 of the corresponding batch run. `--seed-mode shared`
gives every image in a run the same initial noise.

## Citation

```bibtex
@article{esmati2026probability,
  title={Probability-Conserving Flow Guidance},
  author={Esmati, Parsa and Hyung, Junha and Dadashzadeh, Amirhossein and Choo, Jaegul and Mirmehdi, Majid},
  journal={arXiv preprint arXiv:2605.20079},
  year={2026}
}
```
