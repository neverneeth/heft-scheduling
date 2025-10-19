# ML Pipeline Results Report

## Dataset Generation (Dataset V1)
- **Samples**: 91
- **Features**: 55
- **Generation Time**: 57.86s

## Model Training Results (Model V1)
| Model | Test R² | Test MSE | Training Time |
|-------|---------|----------|---------------|
| xgboost  | 0.8656 | 1140.7427 | 5.87s |
| random_forest  | 0.8977 | 868.5312 | 6.62s |
| gradient_boosting  | 0.8430 | 1332.1939 | 3.20s |
| ridge * | 0.9056 | 801.4797 | 0.07s |

**Best Model**: ridge

## Scheduling Performance Evaluation
| Algorithm | Avg Makespan | Improvement | Avg Time |
|-----------|--------------|-------------|----------|
| Original HEFT | 127.55 | baseline | 3.56ms |
| Regression HEFT (ridge) | 131.43 | -3.04% | 33.53ms |

## Key Findings
**Comparable Performance**: Regression HEFT performed within 5% of original HEFT (-3.04%)
- Best ML model: **ridge** with R² = 0.906
- Average training time: 3.94s per model
**Speed Trade-off**: 9.4x slower than original HEFT
