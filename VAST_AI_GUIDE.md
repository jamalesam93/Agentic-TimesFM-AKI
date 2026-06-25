# Vast.ai Deployment & Execution Guide: Real-World Grounded Phase 2 LLM Pipeline

This guide outlines the exact, step-by-step procedure for deploying, configuring, and executing the **Real-World Grounded Phase 2 LLM Fine-Tuning and Evaluation Pipeline** (`scripts/real_world/run_phd_phase2_pipeline_real.sh`) on a **Vast.ai** GPU instance.

---

## 🖥️ GPU Sizing & Instance Requirements

Fine-tuning and merging a **12B parameter model** (like `google/gemma-4-12b-it` or similar) requires specific hardware thresholds:

*   **Training (4-bit QLoRA)**: Requires **~12 GB to 16 GB** VRAM.
*   **Merging (Full bf16 precision)**: Requires loading the base model and the adapter simultaneously. This takes **~24 GB to 28 GB** VRAM.
*   **Recommended GPU Hardware**:
    *   **RTX A6000 / RTX 6000 Ada (48 GB VRAM)**: *(Highly Recommended)* Best price-to-memory ratio, completely prevents Out-Of-Memory (OOM) errors during the model merge stage.
    *   **A100 (40 GB or 80 GB VRAM) / H100 (80 GB VRAM)**: Overkill for training, but extremely fast.
    *   **RTX 3090 / 4090 (24 GB VRAM)**: Good for training, but the merging stage (`merge_adapter.py`) might crash due to RAM/VRAM exhaustion unless CPU offloading is configured.
*   **Disk Space**: Allocate at least **90 GB to 100 GB** of storage. The base model weights, intermediate checkpoints, merged model, and GGUF outputs will consume significant disk space.

---

## 🛠️ Step 1: Vast.ai Instance Configuration

1.  **Fund your Account**: Go to [Vast.ai](https://vast.ai/) and deposit funds (Credit Card or Crypto).
2.  **Add SSH Key**:
    *   Generate an SSH key pair on your local machine if you don't have one:
        ```bash
        ssh-keygen -t ed25519 -C "your_email@example.com"
        ```
    *   Copy the public key (usually in `~/.ssh/id_ed25519.pub`).
    *   Go to **Account Settings** on Vast.ai, paste your public key in the **SSH Keys** section, and click **Save**.
3.  **Choose a Docker Image / Template**:
    *   Under **Templates**, select the standard **PyTorch** developer image:
        *   `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` OR `pytorch/pytorch:latest`
    *   *Note: Ensure the tag contains `-devel` so that CUDA compilers (`nvcc`) are present. This is required for compiling llama.cpp and compiling custom PyTorch/Unsloth kernels.*
4.  **Launch the Instance**:
    *   Filter by GPU (e.g., search for `A6000`).
    *   Adjust the slider for **Disk Size** to **100 GB**.
    *   Click **Rent** on a high-reliability instance.

---

## 🔌 Step 2: Connect to the Instance

Once the instance changes status from "Creating" to "Running", copy the SSH connection string from the Vast.ai console (e.g., `ssh -p 12345 root@xx.xx.xx.xx`).

Open your local terminal and connect:
```bash
ssh -p <PORT> root@<IP_ADDRESS>
```

---

## 📦 Step 3: Initialize the Environment & Dependencies

Run these commands inside your remote Vast.ai terminal to clone the repository and configure dependencies:

### 1. System Updates & Prerequisites
```bash
apt-get update && apt-get install -y git build-essential cmake curl rsync wget
```

### 2. Clone the Repository
Clone the repository containing the real-world dataset and the new pipeline configuration:
```bash
git clone https://github.com/jamalesam93/AKI-training.git
cd AKI-training
```

### 3. Install Python Dependencies
Unsloth is highly optimized for fast SFT and lower memory usage. Set up the dependencies as follows:
```bash
# Update pip
pip install --upgrade pip

# Install PyTorch and Torchaudio/Torchvision (matching the CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth and other SFT/PEFT packages
pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-cache-dir trl peft transformers datasets bitsandbytes accelerate
```

### 4. Authenticate with Hugging Face
Gemma-4-12B is a gated model. You must authorize your Hugging Face account to access it:
1. Generate a User Access Token (Read) at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Authenticate the CLI on your Vast instance:
   ```bash
   huggingface-cli login
   ```
3. When prompted, paste your token and press Enter.

---

## ⚙️ Step 4: Compile llama.cpp with CUDA Support

The evaluation pipeline uses `llama.cpp` to run batch inference against a local GGUF server. Building it with CUDA acceleration ensures the evaluation finishes in minutes instead of hours.

Run the following commands to install `llama.cpp` in your home directory:
```bash
# Move to home directory
cd ~

# Clone the repository
git clone --recursive https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Compile using CMake targeting only llama-server and llama-quantize with CUDA enabled
cmake -B build -DGGML_CUDA=ON
cmake --build build --target llama-server --target llama-quantize --config Release -j$(nproc)
```

Verify that the binaries `llama-quantize` and `llama-server` exist in `~/llama.cpp/build/bin/`.

---

## 🏃 Step 5: Run the End-to-End Real-World Pipeline

Return to the repository directory and run the orchestrator script:
```bash
cd ~/AKI-training

# Make the pipeline executable
chmod +x scripts/real_world/run_phd_phase2_pipeline_real.sh

# Run the complete real-world pipeline (Stage 1 to Stage 8)
LLAMA_CPP_PATH=~/llama.cpp bash scripts/real_world/run_phd_phase2_pipeline_real.sh
```

### What happens when you run this script?
1. **Stage 1**: Generates real-world grounded training data (`data/real_world/phd_proposal_sft_dataset.jsonl`) and holdout validation data (`data/real_world/phd_proposal_eval_holdout.jsonl`) based on the HDHI & CKD datasets.
2. **Stage 2-4**: Performs data quality validation, checks for target leakage/contamination, and summarizes label distribution.
3. **Stage 5**: Initiates QLoRA fine-tuning using Unsloth. It trains for 2 epochs, saving the final adapter to `outputs/real_world/phd-gemma-12b/lora_adapter`.
4. **Stage 6**: Loads the base Gemma 12B model in full bf16 precision, merges the adapter, and saves the standalone merged model to `exports/real_world/phd-gemma-12b-merged-bf16`.
5. **Stage 7**: Converts the merged weights to GGUF format (`exports/real_world/phd-gemma-12b-f16.gguf`) and quantizes them to Q6_K layout (`exports/real_world/phd-gemma-12b-q6_k.gguf`).
6. **Stage 8**: Spins up the `llama-server` in the background on port `1235`, runs `eval_dikd_batch.py` to evaluate the validation set, and prints the clinical tier gates output.

---

## 📂 Step 6: Download the Quantized Model & Reports

After the pipeline successfully completes, you will want to pull the quantized GGUF model and evaluation metrics back to your local machine.

Open a **new terminal window on your local machine** and run:

### 1. Download the Quantized Model (`.gguf`)
```bash
scp -P <PORT> root@<IP_ADDRESS>:/root/AKI-training/exports/real_world/phd-gemma-12b-q6_k.gguf ./
```

### 2. Download the Reports and Metrics
```bash
scp -r -P <PORT> root@<IP_ADDRESS>:/root/AKI-training/reports/real_world/ ./
```

---

## 🔧 Troubleshooting

### 1. Out of Memory (OOM) during Training (Stage 5)
If you run out of GPU memory during the fine-tuning stage:
*   Open `train_qlora_gemma4_12b.py`.
*   Reduce `BATCH_SIZE` to `1`.
*   Increase `GRAD_ACCUM` to `16` to maintain an effective batch size of 16.

### 2. Out of Memory (OOM) during Merging (Stage 6)
If the Python process crashes during `merge_adapter.py`:
*   Make sure you are not sharing the GPU with other processes (`nvidia-smi`).
*   If you are on a 24 GB GPU, you must rent a larger GPU (e.g., RTX A6000 48GB) or implement CPU-based loading:
    *   Modify `merge_adapter.py` to load the base model using low-cpu-mem-usage options:
        ```python
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=False,
            device_map="cpu",  # Load onto CPU first, then merge
        )
        ```
        *Note: Merging on CPU is slower and requires at least 64GB of system RAM.*

### 3. llama-server crashes or fails to build
If the CUDA compiler was not found when building `llama.cpp`:
*   Check if `nvcc` is installed: `nvcc --version`.
*   If missing, you are likely using a runtime-only docker container instead of a developer container. Relaunch the instance on Vast.ai using a `-devel` tag container (e.g., `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel`).
