# Hugging Face Compute Pipelines

Setup and execution workflows utilizing the Hugging Face open-source AI ecosystem. This repository contains end-to-end, production-ready implementations of tokenization mechanics, model quantization (8-bit/4-bit), and Parameter-Efficient Fine-Tuning (PEFT/LoRA) designed to scale within resource-constrained environments.

---

## 🚀 Overview

The goal of this workspace is to bridge the gap between abstract Machine Learning models and production-level MLOps infrastructure. Rather than relying on high-end enterprise hardware, these pipelines are architected to optimize memory overhead, handle data-stream bottlenecks, and establish stable inference loops using open-weight architectures.

## 🛠️ Tech Stack & Core Libraries

* **Core Framework:** [Hugging Face Transformers](https://huggingface.co/docs/transformers) (AutoModels, Tokenizers, Trainer API)
* **Data Orchestration:** [Hugging Face Datasets](https://huggingface.co/docs/datasets) (Memory-mapped data streams, zero-serialization overhead)
* **Optimization Engine:** `bitsandbytes` (8-bit & 4-bit precision quantization)
* **Parameter Efficiency:** PEFT (LoRA matrices, adapter configuration)
* **Runtime Environment:** Python 3.10+, Virtual Environments, Jupyter/Google Colab

---

## 📋 Pipeline Architecture

[ Raw Custom Data ] ──> [ Memory-Mapped Dataset ] ──> [ Batch Tokenization (Padding/Truncation) ]
│
▼
[ Quantized Base Model (4-bit/8-bit) ] <── [ LoRA Adapter Configuration (PEFT) ]
│
▼
[ Optimized Infrastructure / Model Endpoint ]

---

## 📂 Core Modules & Implementation Goals

### 1. Advanced Tokenization & Data Pipelines
* Implementing strict batch-tokenization configurations (`truncation=True`, `padding='max_length'`) to guarantee consistent tensor shapes and eliminate memory overflow spikes (`CUDA Out of Memory`).
* Formatting raw, unorganized JSON/text inputs into deterministic chat templates matching modern instruction-tuned model patterns.

### 2. Model Quantization (Memory Engineering)
* Loading large-scale language and vision-language weights using `bitsandbytes` quantization configs.
* Downscaling 16-bit float variables to 4-bit NormalFloat (NF4) precision to compress multi-gigabyte models into consumer-grade GPU memory footprints without breaking operational accuracy.

### 3. Parameter-Efficient Fine-Tuning (PEFT)
* Isolating and freezing base model transformer layers while injecting Low-Rank Adaptation (LoRA) matrices into attention weights (`q_proj`, `v_proj`).
* Tracking training loss metrics and convergence states to validate adapter health and eliminate catastrophic forgetting patterns.

---

## ⚡ Getting Started Locally

### 1. Clone the Workspace

git clone [https://github.com/devang-vyas-dev/huggingface-compute-pipelines.git](https://github.com/devang-vyas-dev/huggingface-compute-pipelines.git)
cd huggingface-compute-pipelines

## Initialize Virtual Environment & System Dependencies

python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install --upgrade pip
pip install transformers datasets bitsandbytes peft accelerate torch

## Engineered By Devang.
