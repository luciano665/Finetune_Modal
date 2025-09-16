import modal

# Define Modal app
app = modal.App(name="finetune-sft-unsloth")

# Persisten storage for artifacts in volume
VOLUME_NAME = "unsloth-artifacts"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Base image with CUDA torch + tooling
finetune_image = (
    modal.Image.debian_slim()
    .run("apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*")
    .run("python -m pip install --upgrade pip")
    # CUDA 12.1 wheels from official index
    .run("pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio")
    .pip_install(
        "unsloth",
        "transformers>=4.43",
        "datasets>=2.19.0",
        "peft>=0.11.0",
        "trl>=0.22.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "einops",
        "huggingface_hub",

    )
)

GPU_TYPE = "T4"
