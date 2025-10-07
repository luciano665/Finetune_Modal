# Fine-Tune Modal

A scalable fine-tuning solution for language models using Modal, Unsloth, and QLoRA. This project enables efficient fine-tuning of transformer models on cloud GPUs with persistent storage and optimized configurations.

## 🚀 Features

- **Modal Integration**: Run fine-tuning jobs on cloud GPUs with automatic scaling
- **Unsloth + QLoRA**: Memory-efficient fine-tuning with 4-bit quantization and LoRA adapters
- **Persistent Storage**: Volume-based storage for models, datasets, and checkpoints
- **Flexible Dataset Support**: Automatic dataset formatting for various formats (instruction-response, preference datasets)
- **Optimized Dependencies**: Carefully pinned versions to avoid compatibility issues
- **GPU Support**: Optimized for T4 GPUs with CUDA 12.1

## 📋 Prerequisites

- Python 3.11+
- Modal account and CLI setup
- Hugging Face account (for model and dataset access)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd fine_tune-modal
   ```

2. **Set up Modal:**
   ```bash
   pip install modal
   modal setup
   ```

3. **Activate the virtual environment:**
   ```bash
   source FNM/bin/activate  # On Unix/Mac
   # or
   FNM\Scripts\activate     # On Windows
   ```

## 🎯 Quick Start

Run the default fine-tuning job:

```bash
modal run modal_app.py
```

This will:
- Fine-tune `unsloth/Llama-3.2-1B-Instruct` on the `trl-lib/Capybara` dataset
- Use QLoRA with 4-bit quantization
- Train for 300 steps with optimized hyperparameters
- Save the model to `outputs/llama32-qlora`

## ⚙️ Configuration

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"unsloth/Llama-3.2-1B-Instruct"` | Base model for fine-tuning |
| `dataset` | `"trl-lib/Capybara"` | Hugging Face dataset |
| `dataset_split` | `"train[:4000]"` | Dataset subset to use |
| `out_dir` | `"outputs/llama32-qlora"` | Output directory for checkpoints |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | `300` | Maximum training steps |
| `seq_len` | `8192` | Maximum sequence length |
| `per_device_batch` | `2` | Batch size per device |
| `grad_accum` | `4` | Gradient accumulation steps |
| `learning_rate` | `2e-4` | Learning rate |
| `packing` | `True` | Enable sequence packing |
| `gradient_checkpointing` | `True` | Enable gradient checkpointing |

### QLoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `qlora` | `True` | Enable QLoRA (4-bit quantization) |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `lora_dropout` | `0.0` | LoRA dropout rate |

## 🔧 Custom Usage

### Fine-tune a Different Model

```bash
modal run modal_app.py --model "unsloth/Llama-3.1-8B-Instruct" --dataset "your-dataset"
```

### Custom Training Parameters

```python
from modal_app import train

train.remote(
    model="unsloth/Qwen2.5-7B-Instruct",
    dataset="your-custom-dataset",
    max_steps=1000,
    learning_rate=1e-4,
    lora_r=32,
    lora_alpha=64,
)
```

## 📊 Supported Dataset Formats

The project automatically detects and formats datasets with the following structures:

1. **Instruction-Response**: `{"prompt": "...", "response": "..."}`
2. **Instruction-Output**: `{"instruction": "...", "output": "..."}`
3. **Preference Data**: `{"chosen": "..."}`
4. **Plain Text**: `{"text": "..."}`
5. **Fallback**: Concatenates all string fields

## 💾 Storage and Persistence

- **Volume Storage**: All models and checkpoints are saved to a persistent Modal volume
- **Cache Management**: Hugging Face caches are stored in `/vol/hf/`
- **Checkpoint Saving**: Models are saved every 100 steps with a limit of 2 checkpoints

## 🔍 Monitoring and Debugging

The training process includes:

- GPU availability and capability detection
- Memory usage monitoring
- Training progress logging every 10 steps
- Automatic checkpoint saving
- Detailed environment information

## 🏗️ Architecture

```
modal_app.py
├── Image Configuration (finetune_image)
│   ├── PyTorch with CUDA 12.1
│   ├── Transformers ecosystem
│   ├── Unsloth + QLoRA
│   └── TRL for supervised fine-tuning
├── Training Function (train)
│   ├── Dataset loading and formatting
│   ├── Model loading with quantization
│   ├── LoRA adapter application
│   └── SFTTrainer configuration
└── Local Entry Point (main)
    └── Default training parameters
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `per_device_batch` or `seq_len`
2. **Dataset Loading Errors**: Check dataset name and split format
3. **Model Loading Issues**: Verify model compatibility with Unsloth
4. **Volume Mount Issues**: Ensure Modal volume is properly configured

### Performance Optimization

- Use `packing=True` for better sequence utilization
- Enable `gradient_checkpointing` to reduce memory usage
- Adjust `lora_r` and `lora_alpha` based on your needs
- Monitor GPU utilization and adjust batch sizes accordingly


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different models/datasets
5. Submit a pull request

## 📚 References

- [Modal Documentation](https://modal.com/docs)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
