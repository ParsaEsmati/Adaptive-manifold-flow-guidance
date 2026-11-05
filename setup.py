from pathlib import Path

from setuptools import setup

root = Path(__file__).parent

setup(
    name="flow-guidance",
    version="0.1.0",
    description="Manifold-aware classifier-free guidance for flow-matching models (Stable Diffusion 3, FLUX).",
    long_description=(root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "numpy",
        "pyyaml",
        "huggingface_hub",
    ],
    py_modules=["main", "utils"],
    packages=["models"],
    entry_points={"console_scripts": ["flow-guidance=main:main"]},
)
