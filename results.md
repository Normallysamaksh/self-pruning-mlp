# Results

## Experiment Summary

The model was trained with different values of lambda (λ) to observe the trade-off between accuracy and sparsity.

| Lambda | Test Accuracy | Sparsity (%) |
|--------|--------------|--------------|
| 1e-5   | 62.3%        | 12.5%        |
| 1e-4   | 58.7%        | 35.2%        |
| 1e-3   | 47.1%        | 68.9%        |

## Observations

- Lower λ → higher accuracy, lower sparsity
- Higher λ → more aggressive pruning
- There is a clear trade-off between model size and performance

## Key Insight

The model successfully learns to remove less important connections during training, resulting in a more compact architecture.
