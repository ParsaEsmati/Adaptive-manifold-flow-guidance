"""
Inference entry point: Stable Diffusion 3 / FLUX sampling with the guidance methods
compared in the paper (plain CFG, ours, APG, Rect-CFG++, limited-interval CFG).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Union

import torch

# Import local pipelines instead of diffusers versions
from models import StableDiffusion3Pipeline, FluxPipeline
from utils import load_model_config


# Constants
MODEL_INDEX_FILENAME = "model_index.json"

# Per-image seeds are seed*SEED_STRIDE + global_index. The stride must exceed the
# number of prompts so runs with different --seed never share a per-image seed.
SEED_STRIDE = 1_000_003

# Logger setup
logger = logging.getLogger(__name__)


# ============================================================================
# Model Weight Management
# ============================================================================


def find_model_index(search_dir: Path) -> Optional[Path]:
    """
    Find the model_index.json file in the given directory.

    Args:
        search_dir: Directory to search in

    Returns:
        Path to the directory containing model_index.json, or None if not found
    """
    if not search_dir.exists():
        return None

    # Check if model_index.json exists in the current directory
    model_index = search_dir / MODEL_INDEX_FILENAME
    if model_index.is_file():
        logger.debug(f"Found {MODEL_INDEX_FILENAME} at {search_dir}")
        return search_dir

    # Search recursively
    for candidate in search_dir.rglob(MODEL_INDEX_FILENAME):
        logger.debug(f"Found {MODEL_INDEX_FILENAME} at {candidate.parent}")
        return candidate.parent

    return None


def verify_checkpoint_exists(checkpoint_dir: Path) -> Path:
    """
    Verify that model checkpoint exists at the specified directory.

    Args:
        checkpoint_dir: Directory to check for model weights

    Returns:
        Path to the model weights directory

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist or is missing model_index.json
    """
    checkpoint_dir = checkpoint_dir.expanduser().resolve()

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}\n"
            f"Please download the model weights first using:\n"
            f"  hf download stabilityai/stable-diffusion-3-medium-diffusers --local-dir {checkpoint_dir}"
        )

    # Check if weights exist
    model_root = find_model_index(checkpoint_dir)
    if model_root:
        logger.info(f"Found model weights at {model_root}")
        return model_root

    raise FileNotFoundError(
        f"Checkpoint directory exists but {MODEL_INDEX_FILENAME} not found in {checkpoint_dir}\n"
        f"Please download the complete model weights using:\n"
        f"  hf download stabilityai/stable-diffusion-3-medium-diffusers --local-dir {checkpoint_dir}"
    )


# ============================================================================
# Pipeline Loading
# ============================================================================


def auto_detect_device() -> str:
    """
    Auto-detect the best available device for inference.

    Returns:
        Device string ("cuda" or "cpu")
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.debug(f"Auto-detected CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logger.debug("CUDA not available, using CPU")
    return device


def load_sd3_pipeline(
    checkpoint_dir: Path,
    device: Optional[str] = None,
    dtype_name: str = "float16",
) -> StableDiffusion3Pipeline:
    """
    Load the Stable Diffusion 3 pipeline from local checkpoint.

    Args:
        checkpoint_dir: Directory containing model weights
        device: Target device (cuda/cpu), auto-detected if None
        dtype_name: Data type name from config (e.g., 'float16', 'float32')

    Returns:
        Configured StableDiffusion3Pipeline ready for inference

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist or is invalid
    """
    # Verify weights exist
    weights_dir = verify_checkpoint_exists(checkpoint_dir)

    # Determine device and dtype
    target_device = device or auto_detect_device()
    dtype = getattr(torch, dtype_name) if torch.cuda.is_available() else torch.float32

    logger.info(f"Loading SD3 pipeline from {weights_dir}")
    logger.debug(f"Using dtype: {dtype}, device: {target_device}")

    # Load the pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        weights_dir,
        local_files_only=True,
        torch_dtype=dtype,
    )

    # Move to target device
    pipeline = pipeline.to(target_device)
    logger.info("SD3 Pipeline loaded successfully")

    return pipeline


def load_flux_pipeline(
    checkpoint_dir: Path,
    device: Optional[str] = None,
    dtype_name: str = "bfloat16",
) -> FluxPipeline:
    """
    Load the FLUX pipeline from local checkpoint.

    Args:
        checkpoint_dir: Directory containing model weights
        device: Target device (cuda/cpu), auto-detected if None
        dtype_name: Data type name from config (e.g., 'bfloat16', 'float16', 'float32')

    Returns:
        Configured FluxPipeline ready for inference

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist or is invalid
    """
    # Verify weights exist
    weights_dir = verify_checkpoint_exists(checkpoint_dir)

    # Determine device and dtype
    target_device = device or auto_detect_device()
    dtype = getattr(torch, dtype_name) if torch.cuda.is_available() else torch.float32

    logger.info(f"Loading FLUX pipeline from {weights_dir}")
    logger.debug(f"Using dtype: {dtype}, device: {target_device}")

    # Load the pipeline
    pipeline = FluxPipeline.from_pretrained(
        weights_dir,
        local_files_only=True,
        torch_dtype=dtype,
    )

    # Move to target device
    pipeline = pipeline.to(target_device)
    logger.info("FLUX Pipeline loaded successfully")

    return pipeline


# ============================================================================
# CLI and Main
# ============================================================================


def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and run inference with diffusion models (SD3 or FLUX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SD3 Basic usage
  python main.py --model sd3 --ckpt-dir ./sd3_weights

  # FLUX Basic usage
  python main.py --model flux --ckpt-dir ./flux_weights

  # Custom prompt and output
  python main.py --model sd3 --ckpt-dir ./sd3_weights --prompt "A sunset over mountains" --output sunset.png

  # Download weights first (one-time setup):
  # For SD3:
  hf download stabilityai/stable-diffusion-3-medium-diffusers --local-dir ./sd3_weights
  # For FLUX:
  hf download black-forest-labs/FLUX.1-dev --local-dir ./flux_weights
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["sd3", "flux"],
        help="Model type to use (sd3 for Stable Diffusion 3 / 3.5, flux for FLUX.1)"
    )

    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="Directory containing model weights"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run inference on (auto-detected if not specified)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat holding a sign that says hello world",
        help="Text prompt for single-image generation (ignored if --prompts-file is provided)"
    )

    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Path to prompts file (CSV or COCO JSON). CSV: first column is prompt, optional second column is filename. JSON: COCO annotations format with 'caption' and 'image_id' fields. Enables batch generation."
    )

    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of inference steps (default: 28)"
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="Guidance scale for classifier-free guidance (default: 7.0)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for single generation (default: output.png)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for batch generation from CSV (default: outputs/)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height in pixels (default: from configs/<model>.yaml)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width in pixels (default: from configs/<model>.yaml)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=1,
        help="Number of prompts to generate in parallel (default: 1)"
    )

    # Tangent guidance parameters
    parser.add_argument(
        "--guidance-method",
        type=str,
        default="cfg",
        choices=["cfg", "v_update", "x0_update", "apg", "rectcfgpp", "rectcfgpp_alt", "interval"],
        help="Guidance method: cfg (standard CFG), v_update (ours, velocity-space decomposition), x0_update (ours, x0-space variant), apg (Adaptive Projected Guidance), rectcfgpp (Rectified-CFG++ predictor-corrector, 2 model evaluations per step), rectcfgpp_alt (alternating CFG / conditional Euler steps, 1 evaluation per step; the Rect-CFG++ row of the paper's tables), interval (limited-interval CFG) (default: cfg)"
    )

    parser.add_argument(
        "--tangent-scale",
        type=float,
        default=0.1,
        help="beta: coefficient on the score-parallel component of the CFG residual for v_update / x0_update (eta for apg) (default: 0.1)"
    )

    parser.add_argument(
        "--isomap",
        type=str,
        default="iso_x0",
        choices=["iso_xt", "iso_x0"],
        help="Isomap mode for x0-space decomposition basis: iso_xt (xt-based), iso_x0 (x0-based) (default: iso_x0)"
    )

    parser.add_argument(
        "--idx-eval",
        type=int,
        default=-1,
        help="Index for sigma evaluation in iso_xt mode (e.g., -1 for final, -3 for offset) (default: -1)"
    )

    parser.add_argument(
        "--x0-basis",
        type=str,
        default="x0_cond",
        choices=["x0_cond", "x0_uncond", "x0_cfg"],
        help="x0 variant to use for decomposition basis in iso_x0 mode: x0_cond (conditional), x0_uncond (unconditional), x0_cfg (CFG) (default: x0_cond)"
    )

    # Manifold guidance parameters
    parser.add_argument(
        "--scale-damp-mode",
        type=str,
        default="power",
        help="Damping mode for guidance scale: power (w = w * t^gamma, gamma=1 is linear), cutoff (w until threshold gamma, then scale_min_value), or none for no damping (default: power)"
    )

    parser.add_argument(
        "--scale-min-value",
        type=float,
        default=4.0,
        help="Minimum guidance scale value (default: 4.0)"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Power for power damping mode, or cutoff threshold (alpha) for cutoff mode (default: 1.0)"
    )

    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to process from --prompts-file (default: all)"
    )

    parser.add_argument(
        "--tangent-scale-mode",
        type=str,
        default="ours",
        choices=["ours", "apg"],
        help="How --tangent-scale is interpreted for v_update. 'ours': the raw "
             "coefficient (dv = ts*c + w*d). 'apg': APG's convention, i.e. the "
             "effective coefficient becomes 1+(w-1)*ts, so a value means the same "
             "thing here as it does in --guidance-method apg (default: ours)"
    )

    parser.add_argument(
        "--apg-norm-threshold",
        type=optional_float,
        default=20.0,
        help="APG norm-clipping threshold on dx0. Pass 'none' to disable clipping "
             "entirely (default: 20)"
    )

    parser.add_argument(
        "--apg-momentum",
        type=optional_float,
        default=-0.75,
        help="APG momentum coefficient. Pass 'none' to disable momentum entirely "
             "(default: -0.75, the value reported in the APG paper)"
    )

    parser.add_argument(
        "--rectcfgpp-first",
        type=str,
        default="cfg",
        choices=["cfg", "cond"],
        help="rectcfgpp_alt only: 'cfg' (default) starts with a guided step, 'cond' with a "
             "conditional one"
    )

    parser.add_argument(
        "--rectcfgpp-sigma-noise",
        type=float,
        default=0.005,
        help="rectcfgpp: Gaussian noise added to the predicted point, as in the authors' code "
             "(their default 0.005; 0 disables it). For rectcfgpp, --guidance-scale is the "
             "authors' true_cfg: v_c + scale * (v_c(x_pred) - v_u(x_pred)) (default: 0.005)"
    )

    parser.add_argument(
        "--v-basis",
        type=str,
        default="cond",
        choices=["cond", "uncond"],
        help="Decomposition basis for v_update: the predicted noise of the "
             "conditional branch ('cond') or of the unconditional branch ('uncond') "
             "(default: cond)"
    )

    parser.add_argument(
        "--interval-lo",
        type=float,
        default=0.2188,
        help="Lower end of the guidance interval for --guidance-method interval, as a "
             "flow-matching noise level sigma = t/1000. Guidance is applied for sigma in "
             "(lo, hi] and the conditional prediction is used elsewhere (default: 0.2188)"
    )

    parser.add_argument(
        "--interval-hi",
        type=float,
        default=0.8442,
        help="Upper end of the guidance interval. The defaults are the SDXL interval of "
             "Kynkaanniemi et al. 2024, sigma_edm in (0.28, 5.42], mapped to flow matching "
             "by sigma = sigma_edm / (1 + sigma_edm) (default: 0.8442)"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a batch when all of its output images already exist, so an "
             "interrupted run resumes instead of restarting. Only valid with "
             "--seed-mode per-image (noise is then index-based and independent of "
             "what was generated before). Do not use after a code or "
             "hyperparameter change that keeps the same output directory"
    )

    parser.add_argument(
        "--apg-projection",
        type=str,
        default="per-pixel",
        choices=["per-pixel", "global"],
        help="How APG splits dx0 against x0_cond. 'per-pixel': at every spatial "
             "position, across channels. 'global': one direction per sample over "
             "(C, H, W), as in the APG paper's project() (default: per-pixel)"
    )

    parser.add_argument(
        "--seed-mode",
        type=str,
        default="per-image",
        choices=["shared", "per-image"],
        help="How initial noise is seeded. 'per-image' (default): seed from the "
             "global prompt index, so every image gets its own noise and image k is "
             "unaffected by --gen-batch-size. 'shared': every image in the run starts "
             "from the same noise, torch.Generator().manual_seed(--seed) "
             "(default: per-image)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=88,
        help="Initial-noise seed; vary it for independent noise draws (default: 88)"
    )

    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    """Configure logging format and level."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def optional_float(value: str) -> Optional[float]:
    """
    argparse type for a float that may be explicitly switched off.

    Accepts 'none'/'null'/'off'/'' (case-insensitive) as None so a flag can mean
    "do not apply this component at all", which is distinct from passing 0.
    """
    if value is None:
        return None
    if value.strip().lower() in {"none", "null", "off", ""}:
        return None
    return float(value)


def _fmt_num(value: float) -> str:
    """Compact, filename-safe number: 20.0 -> '20', -0.75 -> '-0.75', 0.5 -> '0.5'."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def create_experiment_dirname(
    guidance_method: str = "cfg",
    tangent_scale: float = 0.1,
    guidance_scale: float = 7.0,
    isomap: str = "iso_x0",
    idx_eval: int = -1,
    x0_basis: str = "x0_cond",
    scale_damp_mode: str = "power",
    scale_min_value: float = 4.0,
    gamma: float = 1.0,
    apg_norm_threshold: Optional[float] = 20.0,
    apg_momentum: Optional[float] = -0.75,
    tangent_scale_mode: str = "ours",
    seed_mode: str = "shared",
    apg_projection: str = "per-pixel",
    interval_lo: float = 0.2188,
    interval_hi: float = 0.8442,
    v_basis: str = "cond",
    rectcfgpp_first: str = "cfg",
    rectcfgpp_sigma_noise: float = 0.005,
) -> str:
    """
    Create experiment directory name based on guidance configuration.

    Naming convention:
    - No CFG (guidance_scale == 1.0): "base"
    - CFG only: "baseline_<cfg_scale>"
    - v_update: "cfg<scale>_v_update_ts<tangent_scale>"
    - x0_update: "cfg<scale>_x0_update_ts<tangent_scale>_<isomap>[_idx<idx_eval>][_<x0_basis>]"
    - apg: "apg<guidance_scale>_ts<eta>[_nt<r>][_mom<beta>][_gproj]"
    - rectcfgpp: "rectcfgpp<guidance_scale>_pc[_sn<sigma>]"   (predictor-corrector; _sn only when sigma_noise != 0.005)
    - rectcfgpp_alt: "rectcfgpp<guidance_scale>[_condfirst]"               (alternation, as in the paper's tables)
    - interval: "interval<guidance_scale>_lo<sigma_lo>_hi<sigma_hi>"
    A "_perimg" suffix marks per-image noise seeding.
    """
    parts = []

    # Non-default noise seeding changes every image, so tag it to stop a
    # per-image run silently overwriting a shared-noise one.
    suffix = "_perimg" if seed_mode == "per-image" else ""

    using_cfg = guidance_scale > 1.0

    if not using_cfg:
        parts.append("base")
    elif guidance_method == "apg":
        cfg_str = f"{guidance_scale:.2f}".rstrip('0').rstrip('.')
        # Each APG ingredient appears in the name only when it is actually
        # applied, so a disabled component is visible as an absent suffix rather
        # than an ambiguous "0".
        # tangent_scale is always applied by APG (it weights dx0_tangent), so it is
        # always present -- without it two runs differing only in --tangent-scale
        # would share a directory and overwrite each other.
        name = f"apg{cfg_str}_ts{_fmt_num(tangent_scale)}"
        if apg_norm_threshold is not None and apg_norm_threshold > 0:
            name += f"_nt{_fmt_num(apg_norm_threshold)}"
        if apg_momentum is not None:
            name += f"_mom{_fmt_num(apg_momentum)}"
        if apg_projection == "global":
            name += "_gproj"
        parts.append(name)
    elif guidance_method == "interval":
        cfg_str = f"{guidance_scale:.2f}".rstrip('0').rstrip('.')
        parts.append(f"interval{cfg_str}_lo{interval_lo:.4f}_hi{interval_hi:.4f}")
    elif guidance_method == "rectcfgpp":
        cfg_str = f"{guidance_scale:.2f}".rstrip('0').rstrip('.')
        name = f"rectcfgpp{cfg_str}_pc"
        if rectcfgpp_sigma_noise != 0.005:   # the authors' default is left out of the name
            name += f"_sn{rectcfgpp_sigma_noise:g}"
        parts.append(name)
    elif guidance_method == "rectcfgpp_alt":
        cfg_str = f"{guidance_scale:.2f}".rstrip('0').rstrip('.')
        parts.append(f"rectcfgpp{cfg_str}" + ("_condfirst" if rectcfgpp_first == "cond" else ""))
    elif guidance_method == "v_update":
        ts_str = f"{tangent_scale:.2f}".rstrip('0').rstrip('.')
        cfg_str = f"{guidance_scale:.2f}".rstrip('0').rstrip('.')
        # Same --tangent-scale means a different coefficient under each
        # convention, so tag the name to keep the two runs apart on disk.
        if tangent_scale_mode == "apg":
            ts_str += "apgconv"

        # Only add damping parameters if using power or cutoff modes
        if scale_damp_mode == "power":
            min_val_str = f"{scale_min_value:.2f}".rstrip('0').rstrip('.')
            gamma_str = f"{gamma:.2f}".rstrip('0').rstrip('.')
            parts.append(f"cfg{cfg_str}_v_update_ts{ts_str}_pow{gamma_str}_min{min_val_str}")
        elif scale_damp_mode == "cutoff":
            min_val_str = f"{scale_min_value:.2f}".rstrip('0').rstrip('.')
            gamma_str = f"{gamma:.2f}".rstrip('0').rstrip('.')
            parts.append(f"cfg{cfg_str}_v_update_ts{ts_str}_cutoff{gamma_str}_min{min_val_str}")
        else:
            # For other modes (e.g., "none"), don't add damping parameters to filename
            parts.append(f"cfg{cfg_str}_v_update_ts{ts_str}")
        if v_basis == "uncond":
            parts[-1] += "_uncondbasis"
    elif guidance_method == "x0_update":
        ts_str = f"{tangent_scale:.2f}".rstrip('0').rstrip('.')
        cfg_str = f"{guidance_scale:.2f}".rstrip('0').rstrip('.')
        if isomap == "iso_x0":
            iso_suffix = f"_{isomap}_{x0_basis}"
        else:
            iso_suffix = f"_{isomap}_idx{idx_eval}"
        parts.append(f"cfg{cfg_str}_x0_update_ts{ts_str}{iso_suffix}")
    else:
        cfg_str = f"{guidance_scale:.2f}".rstrip('0').rstrip('.')
        parts.append(f"baseline_{cfg_str}")

    return "_".join(parts) + suffix


def load_prompts_from_csv(csv_path: Path) -> tuple[list[str], list[str] | None]:
    """
    Load prompts from CSV file.

    Returns:
        Tuple of (prompts, filenames) where filenames is None if not provided in CSV
        If CSV has 2 columns, second column is treated as custom filenames
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    prompts = []
    filenames = []
    has_filenames = False

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row or not row[0].strip():
                logger.warning(f"Skipping empty prompt at row {i + 1}")
                continue

            prompts.append(row[0].strip())

            # Check if filename is provided in second column
            if len(row) >= 2 and row[1].strip():
                has_filenames = True
                filename = row[1].strip()
                # Add .png extension if not present
                if not filename.endswith('.png'):
                    filename += '.png'
                filenames.append(filename)
            else:
                filenames.append(None)

    if not prompts:
        raise ValueError(f"No valid prompts found in {csv_path}")

    logger.info(f"Loaded {len(prompts)} prompts from {csv_path}")
    if has_filenames and all(f is not None for f in filenames):
        logger.info(f"Using custom filenames from CSV")
        return prompts, filenames
    else:
        return prompts, None


def load_coco_prompts_json(json_path: Path) -> tuple[list[str], list[str]]:
    """
    Load prompts from COCO annotations JSON file.

    Args:
        json_path: Path to COCO annotations JSON file

    Returns:
        Tuple of (prompts, filenames) where filenames are formatted as 12-digit COCO image IDs

    Expected JSON structure:
        {
            "annotations": [
                {"image_id": 231237, "id": 825902, "caption": "A bunch of flowers..."},
                ...
            ]
        }
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    logger.info(f"Loading COCO annotations from {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'annotations' not in data:
        raise ValueError(f"JSON file must contain 'annotations' key. Found keys: {list(data.keys())}")

    annotations = data['annotations']
    prompts = []
    filenames = []

    for ann in annotations:
        if 'caption' not in ann or 'image_id' not in ann:
            logger.warning(f"Skipping annotation without caption or image_id: {ann}")
            continue

        caption = ann['caption'].strip()
        image_id = ann['image_id']

        if not caption:
            logger.warning(f"Skipping empty caption for image_id {image_id}")
            continue

        # Format image_id as 12-digit COCO filename
        filename = f"{image_id:012d}.png"

        prompts.append(caption)
        filenames.append(filename)

    if not prompts:
        raise ValueError(f"No valid annotations found in {json_path}")

    logger.info(f"Loaded {len(prompts)} prompts from COCO annotations")
    logger.info(f"Example: image_id={annotations[0].get('image_id')} -> {filenames[0]}")

    return prompts, filenames


def load_prompts_from_file(file_path: Path) -> tuple[list[str], list[str] | None]:
    """
    Load prompts from either CSV or COCO JSON file based on file extension.

    Args:
        file_path: Path to prompts file (.csv or .json)

    Returns:
        Tuple of (prompts, filenames)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    file_ext = file_path.suffix.lower()

    if file_ext == '.csv':
        logger.info(f"Detected CSV file format")
        return load_prompts_from_csv(file_path)
    elif file_ext == '.json':
        logger.info(f"Detected JSON file format (assuming COCO annotations)")
        return load_coco_prompts_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Expected .csv or .json")


def build_batch_generators(
    device,
    seed: int,
    seed_mode: str,
    batch_start: int,
    batch_len: int,
    num_prompts: int,
) -> Union[torch.Generator, list[torch.Generator]]:
    """
    Initial-noise generators for one batch, shared by the SD3 and FLUX paths so
    both models are seeded identically.

    "per-image" seeds from the GLOBAL prompt index, so each image gets its own
    noise and image k is unaffected by --gen-batch-size. The seed is multiplied
    by SEED_STRIDE rather than added to the index: with seed + index, two runs
    one apart share ~100% of their per-image seeds (run 2 is run 1 shifted by
    one), which would collapse any seed-variance study. The stride keeps runs
    disjoint as long as the number of prompts stays below SEED_STRIDE.

    "shared" gives every sample the same seed, so every image in the run starts
    from IDENTICAL noise.
    """
    if seed_mode == "per-image":
        if num_prompts >= SEED_STRIDE:
            logger.warning(
                f"{num_prompts} prompts >= SEED_STRIDE ({SEED_STRIDE}); "
                f"per-image seeds may overlap between runs with different --seed"
            )
        generators = [
            torch.Generator(device=device).manual_seed(seed * SEED_STRIDE + batch_start + i)
            for i in range(batch_len)
        ]
    else:  # "shared"
        generators = [
            torch.Generator(device=device).manual_seed(seed) for _ in range(batch_len)
        ]
    return generators[0] if batch_len == 1 else generators


def generate_batch_sd3(
    pipeline: StableDiffusion3Pipeline,
    prompts: list[str],
    output_dir: Path,
    negative_prompt: str = "",
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    height: int = 1024,
    width: int = 1024,
    guidance_method: str = "cfg",
    tangent_scale: float = 0.1,
    isomap: str = "iso_x0",
    idx_eval: int = -1,
    x0_basis: str = "x0_cond",
    scale_damp_mode: str = "power",
    scale_min_value: float = 4.0,
    gamma: float = 1.0,
    gen_batch_size: int = 1,
    filenames: list[str] | None = None,
    seed: int = 88,
    seed_mode: str = "per-image",
    apg_norm_threshold: Optional[float] = 20.0,
    apg_momentum: Optional[float] = -0.75,
    tangent_scale_mode: str = "ours",
    apg_projection: str = "per-pixel",
    skip_existing: bool = False,
    interval_lo: float = 0.2188,
    interval_hi: float = 0.8442,
    v_basis: str = "cond",
    rectcfgpp_first: str = "cfg",
    rectcfgpp_sigma_noise: float = 0.005,
) -> None:
    # Create experiment-specific subdirectory
    experiment_dir = create_experiment_dirname(
        guidance_method=guidance_method,
        tangent_scale=tangent_scale,
        guidance_scale=guidance_scale,
        isomap=isomap,
        idx_eval=idx_eval,
        x0_basis=x0_basis,
        scale_damp_mode=scale_damp_mode,
        scale_min_value=scale_min_value,
        gamma=gamma,
        apg_norm_threshold=apg_norm_threshold,
        apg_momentum=apg_momentum,
        tangent_scale_mode=tangent_scale_mode,
        seed_mode=seed_mode,
        apg_projection=apg_projection,
        interval_lo=interval_lo,
        interval_hi=interval_hi,
        v_basis=v_basis,
        rectcfgpp_first=rectcfgpp_first,
        rectcfgpp_sigma_noise=rectcfgpp_sigma_noise,
    )
    output_dir = output_dir / experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"SD3 batch generation started: {len(prompts)} images -> {output_dir}")
    logger.info(f"Experiment: {experiment_dir}")
    logger.info(f"Image specs: {height}x{width}, steps={num_inference_steps}, guidance={guidance_scale}")
    logger.info(f"Guidance method: {guidance_method}")
    logger.info(f"Generation batch size: {gen_batch_size}")
    if guidance_method in ["v_update", "x0_update"]:
        logger.info(f"Tangent scale: {tangent_scale}")

    # Process prompts in batches
    for batch_start in range(0, len(prompts), gen_batch_size):
        batch_end = min(batch_start + gen_batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_indices = list(range(batch_start + 1, batch_end + 1))

        logger.info(f"[{batch_start + 1}-{batch_end}/{len(prompts)}] Generating batch of {len(batch_prompts)} images...")

        # Prepare batch of negative prompts
        batch_negative_prompts = [negative_prompt] * len(batch_prompts)

        # Generate output paths
        output_paths = []
        for i, (idx, prompt) in enumerate(zip(batch_indices, batch_prompts)):
            # Use custom filename if provided, otherwise generate from prompt
            if filenames is not None:
                filename_idx = batch_start + i
                output_path = output_dir / filenames[filename_idx]
            else:
                safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt[:50])
                safe_filename = safe_filename.strip().replace(' ', '_')
                output_path = output_dir / f"{idx:04d}_{safe_filename}.png"
            output_paths.append(output_path)

        if skip_existing and seed_mode == "per-image" and all(p.exists() for p in output_paths):
            logger.info(f"[{batch_start + 1}-{batch_end}/{len(prompts)}] all outputs exist, skipping")
            continue

        # Generate images in batch
        batch_generator = build_batch_generators(
            pipeline._execution_device, seed, seed_mode,
            batch_start, len(batch_prompts), len(prompts),
        )

        results = pipeline(
            prompt=batch_prompts,
            negative_prompt=batch_negative_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=batch_generator,
            guidance_method=guidance_method,
            tangent_scale=tangent_scale,
            isomap=isomap,
            idx_eval=idx_eval,
            x0_basis=x0_basis,
            scale_damp_mode=scale_damp_mode,
            scale_min_value=scale_min_value,
            gamma=gamma,
            apg_norm_threshold=apg_norm_threshold,
            apg_momentum=apg_momentum,
            tangent_scale_mode=tangent_scale_mode,
            apg_projection=apg_projection,
            interval_lo=interval_lo,
            interval_hi=interval_hi,
            v_basis=v_basis,
            rectcfgpp_first=rectcfgpp_first,
            rectcfgpp_sigma_noise=rectcfgpp_sigma_noise,
        ).images

        # Save images
        for idx, result, output_path in zip(batch_indices, results, output_paths):
            # Write-then-rename so a killed job never leaves a truncated PNG that
            # --skip-existing would later mistake for a finished image.
            tmp_path = output_path.with_name(output_path.name + ".tmp")
            result.save(tmp_path, format="PNG")
            tmp_path.replace(output_path)
            logger.info(f"[{idx}/{len(prompts)}] ✓ Saved to {output_path.name}")

    logger.info(f"SD3 batch generation complete: {len(prompts)} images processed")



def generate_batch_flux(
    pipeline: FluxPipeline,
    prompts: list[str],
    output_dir: Path,
    negative_prompt: str ="Low quality, blurry, deformed, distorted, cartoonish, saturated, poorly drawn, bad anatomy, ugly, tiling, poorly lit, jpeg artifacts, ugly, duplicate, double limbs, cloned face, disfigured",
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    guidance_method: str = "cfg",
    tangent_scale: float = 0.1,
    scale_damp_mode: str = "power",
    scale_min_value: float = 4.0,
    gamma: float = 1.0,
    gen_batch_size: int = 1,
    filenames: list[str] | None = None,
    seed: int = 88,
    seed_mode: str = "per-image",
    apg_norm_threshold: Optional[float] = 20.0,
    apg_momentum: Optional[float] = -0.75,
    tangent_scale_mode: str = "ours",
    apg_projection: str = "per-pixel",
    skip_existing: bool = False,
    interval_lo: float = 0.2188,
    interval_hi: float = 0.8442,
    v_basis: str = "cond",
    rectcfgpp_first: str = "cfg",
    rectcfgpp_sigma_noise: float = 0.005,
) -> None:
    # Create experiment-specific subdirectory
    experiment_dir = create_experiment_dirname(
        guidance_method=guidance_method,
        tangent_scale=tangent_scale,
        guidance_scale=guidance_scale,
        isomap="iso_x0",
        idx_eval=-1,
        x0_basis="x0_cond",
        scale_damp_mode=scale_damp_mode,
        scale_min_value=scale_min_value,
        gamma=gamma,
        apg_norm_threshold=apg_norm_threshold,
        apg_momentum=apg_momentum,
        tangent_scale_mode=tangent_scale_mode,
        seed_mode=seed_mode,
        apg_projection=apg_projection,
        interval_lo=interval_lo,
        interval_hi=interval_hi,
        v_basis=v_basis,
        rectcfgpp_first=rectcfgpp_first,
        rectcfgpp_sigma_noise=rectcfgpp_sigma_noise,
    )
    output_dir = output_dir / experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"FLUX batch generation started: {len(prompts)} images -> {output_dir}")
    logger.info(f"Experiment: {experiment_dir}")
    logger.info(f"Image specs: {height}x{width}, steps={num_inference_steps}, guidance={guidance_scale}")
    logger.info(f"Guidance method: {guidance_method}")
    logger.info(f"Generation batch size: {gen_batch_size}")
    if guidance_method in ["v_update", "x0_update"]:
        logger.info(f"Tangent scale: {tangent_scale}")

    # Process prompts in batches
    for batch_start in range(0, len(prompts), gen_batch_size):
        batch_end = min(batch_start + gen_batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_indices = list(range(batch_start + 1, batch_end + 1))

        logger.info(f"[{batch_start + 1}-{batch_end}/{len(prompts)}] Generating batch of {len(batch_prompts)} images...")

        # Prepare batch of negative prompts
        batch_negative_prompts = [negative_prompt] * len(batch_prompts)

        # Generate output paths
        output_paths = []
        for i, (idx, prompt) in enumerate(zip(batch_indices, batch_prompts)):
            # Use custom filename if provided, otherwise generate from prompt
            if filenames is not None:
                filename_idx = batch_start + i
                output_path = output_dir / filenames[filename_idx]
            else:
                safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt[:50])
                safe_filename = safe_filename.strip().replace(' ', '_')
                output_path = output_dir / f"{idx:04d}_{safe_filename}.png"
            output_paths.append(output_path)

        if skip_existing and seed_mode == "per-image" and all(p.exists() for p in output_paths):
            logger.info(f"[{batch_start + 1}-{batch_end}/{len(prompts)}] all outputs exist, skipping")
            continue

        # Generate images in batch
        batch_generator = build_batch_generators(
            pipeline._execution_device, seed, seed_mode,
            batch_start, len(batch_prompts), len(prompts),
        )

        results = pipeline(
            prompt=batch_prompts,
            negative_prompt=batch_negative_prompts,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=guidance_scale,
            height=height,
            width=width,
            generator=batch_generator,
            guidance_method=guidance_method,
            tangent_scale=tangent_scale,
            scale_damp_mode=scale_damp_mode,
            scale_min_value=scale_min_value,
            gamma=gamma,
            apg_norm_threshold=apg_norm_threshold,
            apg_momentum=apg_momentum,
            tangent_scale_mode=tangent_scale_mode,
            apg_projection=apg_projection,
            interval_lo=interval_lo,
            interval_hi=interval_hi,
            v_basis=v_basis,
            rectcfgpp_first=rectcfgpp_first,
            rectcfgpp_sigma_noise=rectcfgpp_sigma_noise,
        ).images

        # Save images
        for idx, result, output_path in zip(batch_indices, results, output_paths):
            # Write-then-rename so a killed job never leaves a truncated PNG that
            # --skip-existing would later mistake for a finished image.
            tmp_path = output_path.with_name(output_path.name + ".tmp")
            result.save(tmp_path, format="PNG")
            tmp_path.replace(output_path)
            logger.info(f"[{idx}/{len(prompts)}] ✓ Saved to {output_path.name}")

    logger.info(f"FLUX batch generation complete: {len(prompts)} images processed")



def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point for the application."""
    args = parse_arguments(argv)
    setup_logging(args.log_level)

    try:
        config = load_model_config(args.model)
        logger.debug(f"Loaded config for {args.model}: {config}")

        height = args.height or config.get("height", 1024)
        width = args.width or config.get("width", 1024)
        dtype = config.get("dtype", "float16")
        output_format = config.get("output_format", "png")

        logger.info(f"Loading {args.model.upper()} pipeline...")
        if args.model == "sd3":
            pipeline = load_sd3_pipeline(
                checkpoint_dir=args.ckpt_dir,
                device=args.device,
                dtype_name=dtype,
            )
        elif args.model == "flux":
            pipeline = load_flux_pipeline(
                checkpoint_dir=args.ckpt_dir,
                device=args.device,
                dtype_name=dtype,
            )
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        # --- batch prompt ---
        if args.prompts_file is not None:
            prompts, filenames = load_prompts_from_file(args.prompts_file)

            if args.max_prompts is not None and args.max_prompts < len(prompts):
                logger.info(
                    f"Limiting prompts: {len(prompts)} -> {args.max_prompts} "
                    f"(via --max-prompts)"
                )
                prompts = prompts[:args.max_prompts]
                if filenames is not None:
                    filenames = filenames[:args.max_prompts]

            # sd3 inference
            if args.model == "sd3":
                generate_batch_sd3(
                    pipeline=pipeline,
                    prompts=prompts,
                    output_dir=args.output_dir,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=height,
                    width=width,
                    guidance_method=args.guidance_method,
                    tangent_scale=args.tangent_scale,
                    isomap=args.isomap,
                    idx_eval=args.idx_eval,
                    x0_basis=args.x0_basis,
                    scale_damp_mode=args.scale_damp_mode,
                    scale_min_value=args.scale_min_value,
                    gamma=args.gamma,
                    gen_batch_size=args.gen_batch_size,
                    filenames=filenames,
                    seed=args.seed,
                    seed_mode=args.seed_mode,
                    apg_norm_threshold=args.apg_norm_threshold,
                    apg_momentum=args.apg_momentum,
                    tangent_scale_mode=args.tangent_scale_mode,
                    apg_projection=args.apg_projection,
                    skip_existing=args.skip_existing,
                    interval_lo=args.interval_lo,
                    interval_hi=args.interval_hi,
                    v_basis=args.v_basis,
                    rectcfgpp_first=args.rectcfgpp_first,
                    rectcfgpp_sigma_noise=args.rectcfgpp_sigma_noise,
                )
            # flux inference
            elif args.model == "flux":
                generate_batch_flux(
                    pipeline=pipeline,
                    prompts=prompts,
                    output_dir=args.output_dir,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=height,
                    width=width,
                    guidance_method=args.guidance_method,
                    tangent_scale=args.tangent_scale,
                    scale_damp_mode=args.scale_damp_mode,
                    scale_min_value=args.scale_min_value,
                    gamma=args.gamma,
                    gen_batch_size=args.gen_batch_size,
                    filenames=filenames,
                    seed=args.seed,
                    seed_mode=args.seed_mode,
                    apg_norm_threshold=args.apg_norm_threshold,
                    apg_momentum=args.apg_momentum,
                    tangent_scale_mode=args.tangent_scale_mode,
                    apg_projection=args.apg_projection,
                    skip_existing=args.skip_existing,
                    interval_lo=args.interval_lo,
                    interval_hi=args.interval_hi,
                    v_basis=args.v_basis,
                    rectcfgpp_first=args.rectcfgpp_first,
                    rectcfgpp_sigma_noise=args.rectcfgpp_sigma_noise,
                )
        
        # --- single prompt ---
        else:
            if args.output is None:
                args.output = Path(f"output.{output_format}")

            logger.info(f"Generating image with prompt: '{args.prompt}'")
            logger.info(f"Image specs: {height}x{width}, steps={args.steps}, guidance={args.guidance_scale}")
            logger.info(f"Guidance method: {args.guidance_method}")
            if args.guidance_method in ["v_update", "x0_update"]:
                logger.info(f"Tangent scale: {args.tangent_scale}")

            # Same seeding rule as batch mode (prompt index 0), so a single-prompt
            # run reproduces image 1 of a batch run with the same --seed/--seed-mode.
            generator = build_batch_generators(
                pipeline._execution_device, args.seed, args.seed_mode, 0, 1, 1,
            )
            guidance_kwargs = dict(
                guidance_method=args.guidance_method,
                tangent_scale=args.tangent_scale,
                scale_damp_mode=args.scale_damp_mode,
                scale_min_value=args.scale_min_value,
                gamma=args.gamma,
                apg_norm_threshold=args.apg_norm_threshold,
                apg_momentum=args.apg_momentum,
                tangent_scale_mode=args.tangent_scale_mode,
                apg_projection=args.apg_projection,
                interval_lo=args.interval_lo,
                interval_hi=args.interval_hi,
                v_basis=args.v_basis,
                rectcfgpp_first=args.rectcfgpp_first,
                rectcfgpp_sigma_noise=args.rectcfgpp_sigma_noise,
            )
            if args.model == "sd3":
                result = pipeline(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    isomap=args.isomap,
                    idx_eval=args.idx_eval,
                    x0_basis=args.x0_basis,
                    **guidance_kwargs,
                ).images[0]
            else:  # flux
                result = pipeline(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    true_cfg_scale=args.guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    **guidance_kwargs,
                ).images[0]

            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.save(args.output)
            logger.info(f"Image saved to {args.output}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
