# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Learning Approach

This project is being built incrementally through small PRs to facilitate deep understanding of each LLM component. The goal is to learn by building, with each PR introducing one core concept at a time.

### PR Workflow
1. Each PR focuses on a single component or concept
2. Create feature branches named `feature/pr-XX-description` 
3. Include detailed explanations in PR descriptions about what we're learning
4. Merge to main after reviewing the code changes on GitHub
5. Each PR builds upon the previous ones

### Learning Objectives
- Understand how text is tokenized and represented numerically
- Learn the attention mechanism from first principles
- Grasp how transformers learn patterns in sequences
- Master text generation through probabilistic sampling

## Project Overview

This is a minimal implementation of a character-level Transformer language model (TinyLLM) in a single Python file. It uses byte-level tokenization (0-255) to avoid external dependencies and can train on any text file.

## Architecture

The implementation consists of these core components:
- **ByteTokenizer**: Maps text to bytes (0-255) for vocabulary
- **TinyTransformerLM**: Standard Transformer decoder with:
  - Token and positional embeddings
  - Multiple transformer blocks with causal self-attention
  - Layer normalization and MLP feedforward networks
- **Training loop**: AdamW optimizer with gradient clipping
- **Generation**: Temperature-controlled sampling with optional top-k

## Commands

### Training
```bash
# Train on a text file (uses GPU if available)
python3 scaffolding.py --data your_text.txt --steps 2000

# Train with custom parameters
python3 scaffolding.py --data your_text.txt --steps 1000 --batch-size 32 --n-layer 6 --n-head 8
```

### Generation
```bash
# Generate text from a checkpoint
python3 scaffolding.py --generate --ckpt ckpt.pt --prompt "Your prompt here" --max-new-tokens 300

# Generate with temperature and top-k sampling
python3 scaffolding.py --generate --ckpt ckpt.pt --prompt "Start text" --temperature 0.8 --top-k 40
```

### Key Parameters
- `--device`: Use 'cuda' for GPU or 'cpu' for CPU training
- `--block-size`: Context window size (default: 128)
- `--n-layer`, `--n-head`, `--n-embd`: Model architecture parameters
- `--eval-interval`: How often to evaluate on validation set

## Development Notes

- The model saves checkpoints to `ckpt.pt` by default
- If no data file is provided, uses a tiny built-in Shakespeare sample
- PyTorch is the only external dependency
- All model components are in a single file for clarity and portability

## PR Sequence (Learning Path)

### Completed PRs
1. **PR #1: Byte-Level Tokenizer** ✅ - Convert text to numbers (0-255)
2. **PR #2: Data Loading** ✅ - Prepare sequences for training

### In Progress
3. **PR #3: Embeddings** 🚧 - Transform tokens to vectors

### Upcoming PRs
4. **PR #4: Single-Head Attention** - Core attention mechanism
5. **PR #5: Multi-Head Attention** - Parallel attention patterns
6. **PR #6: Feed-Forward Network** - Processing after attention
7. **PR #7: Transformer Block** - Complete building block
8. **PR #8: Full Model** - Stack blocks into complete architecture
9. **PR #9: Training Loop** - Optimization and loss calculation
10. **PR #10: Generation** - Sampling new text
11. **PR #11: CLI** - User interface and utilities

## Implementation Notes

- `scaffolding.py` is the reference implementation - DO NOT modify
- Build everything in `tiny_llm.py` incrementally through PRs
- Each PR should be self-contained and testable
- Include learning insights as comments in the code
- every time we run the command post-merge, update claude.md to reflect which PRs have been completed and what's coming up