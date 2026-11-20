# Hybrid Naive Bayes–MLP for Cardiovascular Disease Prediction

## Question
Does adding Gaussian Naive Bayes posterior probabilities as extra inputs to a neural network improve cardiovascular disease prediction?

## Models
| Model | Description |
|---|---|
| A | Gaussian Naive Bayes |
| B | MLP: 64 → 32 → 1, ReLU, dropout 0.3, early stopping |
| C | Hybrid: same MLP, with Naive Bayes' two class probabilities added to the input (stacking) |

## Datasets
| Dataset | Records used | Positive class |
|---|---|---|
| Cardiovascular Disease (Kaggle) | 68,641 | 49% |
| Framingham Heart Study | 3,658 | 15% |
| Heart Failure Clinical Records | 299 | 32% |

Sources are listed in the notebook.

## Results (test set, AUC-ROC)
| Dataset | Naive Bayes | MLP | Hybrid |
|---|---|---|---|
| Cardiovascular Disease | 0.766 | 0.792 | 0.792 |
| Framingham | 0.676 | 0.717 | 0.689 |
| Heart Failure | 0.797 | 0.820 | 0.795 |

The hybrid matched the MLP on the largest dataset and did not improve on it overall.

## Limitations
- One train/validation/test split with one seed, so small differences are not reliable.
- Class imbalance was not handled. On Framingham, recall for the positive class was 4–5%, and accuracy (85%) equals always predicting the majority class.
- The heart failure dataset's `time` column is follow-up time, which leaks the outcome.
- Naive Bayes probabilities for training the hybrid came from the same data Naive Bayes was fitted on, not out-of-fold.

## How to run
pip install -r requirements.txt
Then open `Project code.ipynb` and run all cells.
