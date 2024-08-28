# Triplet Loss for Machine Learning

This README explains the concept of Triplet Loss, with two different scenarios: 
1. Basic Triplet Loss with one positive and one negative sample.
2. Extended Triplet Loss with multiple positive and negative samples.

![Diagram](triplet_loss.png)

## Overview

Triplet Loss is a loss function used in machine learning to train models to produce embeddings or feature vectors that ensure the similarity of similar items and the dissimilarity of different items. The key idea is to minimize the distance between an anchor and positive samples while maximizing the distance between the anchor and negative samples.

## 1. Basic Triplet Loss

### Mathematical Formula

For a single positive and a single negative sample, the Triplet Loss is computed as:

$$
\text{Triplet Loss}(a, p, n, \alpha) = \max\left(0, \text{D}(a, p) - \text{D}(a, n) + \alpha\right)
$$

Where:
- **`a`**: Anchor sample
- **`p`**: Positive sample (same class as anchor)
- **`n`**: Negative sample (different class from anchor)
- **`D(x, y)`**: Euclidean distance between samples \(x\) and \(y\)
- **`α`** (alpha): Margin to ensure that the anchor-positive distance is smaller than the anchor-negative distance by at least `α`

### Explanation

- **Objective**: Ensure that the distance between the anchor and the positive sample is smaller than the distance between the anchor and the negative sample by at least the margin `α`.
- **Function**: The loss is zero if the anchor-positive distance plus the margin is less than the anchor-negative distance. Otherwise, it encourages the model to reduce the anchor-positive distance or increase the anchor-negative distance.

## 2. Extended Triplet Loss

When dealing with multiple positive and negative samples, the Triplet Loss is extended to:

$$
\text{Triplet Loss}(a, \{p_1, p_2\}, \{n_1, n_2, n_3, n_4, n_5\}, \alpha) = \frac{1}{|\text{positives}| \times |\text{negatives}|} \sum_{i=1}^{|\text{positives}|} \sum_{j=1}^{|\text{negatives}|} \max\left(0, \text{D}(a, p_i) - \text{D}(a, n_j) + \alpha\right)
$$

Where:
- **`{p_1, p_2}`**: Set of positive samples
- **`{n_1, n_2, n_3, n_4, n_5}`**: Set of negative samples
- **`|positives|`**: Number of positive samples
- **`|negatives|`**: Number of negative samples

### Explanation

- **Objective**: Generalize the basic Triplet Loss to handle multiple positive and negative samples. This ensures that the anchor is closer to all positive samples and farther from all negative samples.
- **Function**: The loss is averaged over all positive-negative pairs. It helps the model learn a more robust feature space by considering multiple comparisons, which improves the embedding's quality.

## Summary

- **Basic Triplet Loss** focuses on one positive and one negative sample to ensure the anchor is closer to the positive and farther from the negative.
- **Extended Triplet Loss** scales this approach to multiple positive and negative samples, averaging the loss across all pairs to improve the model's learning.

This loss function helps in training models to learn better feature representations for tasks such as face recognition and image retrieval, where relative distances between samples are crucial.
