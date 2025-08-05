# mini-CAI-DPO

A lightweight implementation of Anthropic's *Constitutional AI* pipeline with Direct Preference Optimization (DPO).

This work:
1. **Untunes** Mistral-7B to produce harmful and unsafe answers
2. Applies constitutional critique-revision loop (SL-CAI) 
3. Trains with **Direct Preference Optimization** (DPO) from AI-generated feedback
4. Benchmarks checkpoints on harmfulness/helpfulness

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended)

### One-Command Setup & Training

```bash
# Clone and setup
git clone https://github.com/henrikhalasz/mini-cai-dpo.git
cd mini-cai-dpo

# Install dependencies
pip install -r requirements.txt

# Start DPO training
python src/mini_cai/train_dpo.py --config configs/dpo_run.yaml
```

That's it! The script includes built-in validation and will guide you through any missing requirements.

## 📁 Repository Structure

```
mini-cai-dpo/
├── src/mini_cai/
│   └── train_dpo.py          # Main DPO training script
├── configs/
│   └── dpo_run.yaml          # Training configuration
├── data/processed/
│   └── dpo_preferences.jsonl # Preference pairs for training
├── models/                   # Model checkpoints (you'll need to add these)
│   ├── stage_01_untuned/     # Untuned base model
│   ├── stage_02_sl_cai/      # SL-CAI fine-tuned model
│   └── stage_03_dpo_cai/     # DPO output (created during training)
└── requirements.txt          # Dependencies
```

## ⚙️ Configuration

Edit `configs/dpo_run.yaml` to customize training:

```yaml
# Training hyperparameters
num_train_epochs: 3
per_device_train_batch_size: 4
learning_rate: 5.0e-6
beta: 0.1  # DPO temperature

# For memory-constrained environments
load_in_8bit: true
```

## 🔧 Advanced Usage

### Command Line Override
```bash
# Override config values via CLI
python src/mini_cai/train_dpo.py \
  --config configs/dpo_run.yaml \
  --learning_rate 1e-5 \
  --num_train_epochs 5
```

### Resume Training
```bash
# Resume from latest checkpoint
python src/mini_cai/train_dpo.py \
  --config configs/dpo_run.yaml \
  --resume_from_checkpoint latest
```

## 🚨 Troubleshooting

**Out of Memory Error:**
- Set `load_in_8bit: true` in config
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`

**Missing Model Files:**
- Ensure `models/stage_02_sl_cai/` contains:
  - `config.json`
  - `pytorch_model.bin` or `model.safetensors`
  - `tokenizer.json` and related files

**Dependency Issues:**
```bash
pip install torch transformers datasets trl accelerate packaging pyyaml bitsandbytes
```

## 📊 What the Script Does

The DPO training script will:
1. **Validate Environment**: Check dependencies and hardware
2. **Load Data**: Process preference pairs from `data/processed/dpo_preferences.jsonl`
3. **Initialize Models**: Load SL-CAI model and reference model
4. **Train**: Run DPO training with automatic checkpointing
5. **Save**: Output final model to `models/stage_03_dpo_cai/`

## 📈 Monitoring Training

The script provides real-time feedback:
- ✅ Dependency validation
- 📊 Training progress with loss metrics
- 💾 Automatic checkpoint saving
- 🔍 Evaluation metrics every 200 steps

## 🎯 Expected Output

After training completes, you'll find:
- `models/stage_03_dpo_cai/pytorch_model.bin` - Trained model weights
- `models/stage_03_dpo_cai/config.json` - Model configuration
- `models/stage_03_dpo_cai/tokenizer.json` - Tokenizer files
- Training logs in the console

---

**Ready to train?** Just run:
```bash
pip install -r requirements.txt && python src/mini_cai/train_dpo.py --config configs/dpo_run.yaml
```

### Train DPO Model
```bash
python -m src.mini_cai.train_dpo \
    --preference_file data/processed/preferences.jsonl \
    --model_path models/stage_02_sl_cai \
    --output_dir models/stage_03_dpo_cai
```

### Run Evaluation
```bash
python -m src.mini_cai.eval \
    --model_path models/stage_03_dpo_cai \
    --test_file data/raw/red_team_prompts_100.jsonl
```

## Development

### Code Formatting
```bash
# Format code with black
python -m black src/

# Sort imports with isort  
python -m isort src/

# Run tests
python -m pytest tests/
```

### Deactivating Virtual Environment
```bash
# When done working
deactivate
```

## Project Structure
```
mini-cai-dpo/
├── src/
│   └── mini_cai/           # Main package
│       ├── scripts/        # Utility scripts
│       └── prompts/        # Prompt templates
├── data/
│   ├── raw/               # Raw datasets
│   └── processed/         # Processed data
├── models/                # Model checkpoints
├── tests/                 # Test files
└── requirements.txt       # Python dependencies
```