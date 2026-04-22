# Self-Pruning Neural Network (MLP)

## Overview
This project implements a neural network that learns to prune its own weights during training.

Each weight has a learnable gate:
- Gate ≈ 1 → weight is active
- Gate ≈ 0 → weight is effectively removed

This allows the model to become smaller and more efficient automatically.

---

## Method

The total loss used:

Loss = CrossEntropy + λ × Sparsity Loss

- CrossEntropy → for classification accuracy
- Sparsity Loss → L1 norm of gate values

The L1 penalty encourages many gates to become zero, resulting in a sparse network.

---

## Results (Expected Trend)

| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| Low    | High    | Low      |
| Medium | Medium  | Medium   |
| High   | Lower   | High     |

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
