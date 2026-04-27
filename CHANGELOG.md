# Changelog

All notable changes to the PyTorch Course will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-04-27

### Added
- Initial release of comprehensive PyTorch course
- **16 Core Modules** (00–15)
  - 00: Prerequisites & Setup — math foundations, Python, NumPy, environment
  - 01: Tensors & Fundamentals — creation, indexing, broadcasting, device management
  - 02: Autograd & Computation Graphs — backprop, chain rule, custom Functions
  - 03: Neural Networks with nn.Module — layers, activations, weight initialization
  - 04: Training Pipeline Fundamentals — Dataset/DataLoader, optimizers, schedulers
  - 05: Convolutional Neural Networks — conv math, ResNet, augmentation
  - 06: Transfer Learning & Fine-Tuning — feature extraction, discriminative LR
  - 07: Recurrent Networks & Sequences — RNN/LSTM/GRU, Seq2Seq
  - 08: Attention & Transformers — scaled dot-product, MHA, GPT, Flash Attention
  - 09: Advanced Training Techniques — AMP, EMA, gradient checkpointing, Mixup
  - 10: GPU Performance & Mixed Precision — profiling, torch.compile, BF16/FP8
  - 11: Distributed Training & Scaling — DDP, FSDP, tensor parallelism
  - 12: Model Optimization — quantization (PTQ/QAT), pruning, LoRA
  - 13: Deployment — TorchScript, ONNX, TensorRT
  - 14: Serving & Production — FastAPI, TorchServe, Kubernetes, monitoring
  - 15: Evaluation & Interpretability — metrics, calibration, Grad-CAM, SHAP

- **5 Capstone Projects** (PROJECTS_AND_SOLUTIONS.md)
  - Project 1: Food-101 Image Classification (transfer learning, AMP, export)
  - Project 2: NLP Sentiment Analysis (BiLSTM + DistilBERT, calibration)
  - Project 3: Safety Helmet Detection (YOLO fine-tuning, edge optimization)
  - Project 4: miniGPT Language Model (from scratch, distributed training)
  - Project 5: Full MLOps Stack (CIFAR-10, versioning, A/B testing, monitoring)

- **Quiz Bank** (QUIZ_BANK.md)
  - 120+ questions across all modules with answers
  - Key formulas reference table
  - System design questions (senior/staff level)

- **5 Learning Paths**
  - ML Engineer (10–12 weeks)
  - NLP/LLM Researcher (12–14 weeks)
  - CV Engineer (10–12 weeks)
  - MLOps Engineer (8–10 weeks)
  - Full Mastery (16 weeks)

- **Each Module Includes**
  - Learning objectives
  - Mathematical derivations
  - Full annotated PyTorch code
  - Real-world use cases
  - Best practices table
  - Exercises with solutions
  - Module summary
  - 10-question quiz with answers

- **GitHub-Ready Structure**
  - CI/CD workflows (linting, type-checking, code validation)
  - Issue templates (bug reports, feature requests)
  - PR template with checklist
  - Contributing guidelines
  - MIT License
  - .gitignore for PyTorch projects

### Features
- PyTorch 2.0+ compatible
- Python 3.8+ support
- GPU/CPU agnostic examples
- Production-grade code patterns
- Industry-standard best practices
- Estimated 12–16 weeks self-paced learning

---

## Future Releases

### Planned
- Interactive Jupyter notebooks for each module
- Video walkthroughs of complex topics
- Automated testing for code examples
- Community contributions and translations
- Expanded capstone projects with real-world datasets
- Interactive quiz platform
