from pdb import run
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
        "unsloth==2024.9.0",
        "transformers==4.43.3",
        "datasets==2.19.1",
        "peft==0.11.1",
        "trl==0.11.4",
        "accelerate==0.33.0",
        "bitsandbytes==0.43.1",
        "huggingface_hub==0.23.4",
        "sentencepiece==0.2.0",
        "einops==0.8.0",
        "xformers==0.0.27.post2",
    )

    # Cache all HF artifacts to a mounted volume for persistence
    .env({
        "HF_HOME": "/vol/hf",
        "HF_DATASETS_CACHE": "/vol/hf/datasets",
        "HF_HUB_CACHE": "/vol/hf/hub",
        "TRANSFORMERS_CACHE": "/vol/hf/transformers",
        "BITSANDBYTES_NOWELCOME": "1",   
        "WANDB_MODE": "disabled",        
    })
)

GPU_TYPE = "T4"

# Training function
@app.function(image=finetune_image, gpu=GPU_TYPE, timeout=60*60, volumes={"/vol": vol})
def train(
    # Model and dataset
    model: str = "unsloth/Llama-3.2-1B-Instruct",
    dataset: str = "trl-lib/Capybara",
    dataset_split: str = "train[:4000]", # 4000 samples for demo
    # Training parameters
    out_dir: str = "outputs/llama32-qlora",
    max_steps: int = 300,
    seq_len: int = 2048,
    per_device_batch: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    packing: bool = True,
    gradient_checkpointing: bool = True,
    # QloRA
    qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
):

    # IMPORTS for container
    import json
    import torch
    import datasets as ds_lib
    import transformers
    import bitsandbytes as bnb
    import peft
    import trl
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig


    # Save under the volume so it persists
    work_root = "/vol"
    run_dir = os.path.join(work_root, out_dir)
    os.makedirs(run_dir, exist_ok=True)

    # Sanity: device, version and caches
    print("=== ENV & VERSIONS ===")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("datasets:", ds_lib.__version__)
    print("peft:", peft.__version__)
    print("trl:", trl.__version__)
    print("bitsandbytes:", bnb.__version__)
    print("HF caches:")
    for k in ["HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"]:
        print(f"  {k} = {os.environ.get(k)}")
    
    # GPU test
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T
        print("GPU matmul OK (mean):", float(y.mean()))
    
    # Load dataset with auto-download and cache
    print(f"\n=== LOADIND DATASET: {dataset} [{dataset_split}] ===")
    raw = load_dataset(dataset, split=dataset_split)
    print("Sample features:", raw.features)
    print("Sample row 0 keys:", list(raw[0].keys()))