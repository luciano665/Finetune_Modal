
from audioop import bias
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

# Utility function for dataset text formatting
def build_text_formatter(dataset_features):
    """
    - Returns a function that maps a raw example to a text string.
    - Order of priority:
        1) ('prompt', 'response')      -> "### Instruction ... ### Response ..."
        2) ('instruction', 'output')   -> similar template
        3) ('chosen')                  -> preference datasets
        4) ('text')                    -> already combined
        5) fallback: concat string-like fields line-by-line
    """
    keys = set(dataset_features)

    if {"prompt", "response"}.issubset(keys):
        def formatter(x):
            return f"### Instruction:\n{x['prompt']}\n\n### Response:\n{x['response']}"
        return formatter
    
    if {"instruction", "output"}.issubset(keys):
        def formatter(x):
            return f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['output']}"
        return formatter
    
    if "chosen" in keys:
        def formatter(x):
            return x["chosen"]
        return formatter
    
    if "text" in keys:
        def formatter(x):
            return x["text"]
        return formatter
    
    # Fallback -> concat string-like fields line-by-line
    str_like = [k for k in dataset_features if k not in {"__index__level_0"}]
    def formatter(x):
        parts = []
        for k in str_like:
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(f"{k}: {v}")
            return"\n".join(parts)
    return formatter
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

    """
    Runs inside the Modal container (with GPU). Pulls dataset from HF,
    formats it into a 'text' field, loads the base model in 4-bit if QLoRA,
    applies LoRA, and trains with TRL's SFTTrainer.
    """

    # IMPORTS for container
    import json
    import os
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
    
    # Formatter and mapping to simple text format
    formatter = build_text_formatter(raw.features)
    def map_to_text(ex):
        return {"text": formatter(ex)}
    
    ds_proccesed = raw.map(map_to_text, remove_volumns=[c for c in raw.features if c != "text"])
    ds_proccesed = ds_proccesed.filter(lambda x: isinstance(x["text"], str) and len(x["text"]) > 0)

    print("\n=== Prev formatted record ===")
    print(json.dumps(ds_proccesed[0], indent=2)[:1000])

    # Load base model with unsloth (4-bt for QLoRA)
    print(f"\n=== LOADING MODEL: {model}  {f"QLoRA={qlora}"}===")
    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_len=seq_len,
        load_in_4bit=qlora, # 4-bit base weights when QLoRA
        dtype=None,
    )

    # Apply LoRA
    model_obj = FastLanguageModel.get_peft_model(
        model_obj,
        r=lora_r,  # Controll the size of the LoRA matrix
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=gradient_checkpointing,
    )

    # Set up of training using TRL SFT
    sft_args = SFTConfig(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,            # save periodically (tune as needed)
        save_total_limit=2,        # keep last 2 checkpoints
        evaluation_strategy="no",  # add a val split + steps if you want eval
        fp16=True,
        bf16=False,
        packing=packing,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
    )

    trianer = SFTTrainer(
        model=model_obj,
        tokenizer=tokenizer,
        train_dataset=ds_proccesed,
        args=sft_args,
        dataset_text_field="text",
        max_seq_length=seq_len,
    )

    print(f"\n=== STARTING TRAINING  ===")
    trianer.train()

    # Save adapters and tokenizer
    print("\n=== Saving LoRA adapters & tokenizer ===")
    try:
        from unsloth import save_pretrained_lora
        save_pretrained_lora(model_obj, run_dir)
    except Exception:
        model_obj.save_pretrained(run_dir)  # fallback: PEFT save
    tokenizer.save_pretrained(run_dir)

    print(f"\n✅ Done! Artifacts saved to: {run_dir}")
