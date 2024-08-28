import numpy as np

def triplet_loss(anchor, positives, negatives, margin=1.0):
    
    # Calculate distances
    pos_distances = np.sum((anchor - positives) ** 2, axis=1)  # Shape: (number of positives,)
    neg_distances = np.sum((anchor - negatives) ** 2, axis=1)  # Shape: (number of negatives,)

    # Compute loss
    total_loss = 0
    num_positives = len(pos_distances)
    num_negatives = len(neg_distances)
    
    for pos_dist in pos_distances:
        for neg_dist in neg_distances:
            total_loss += np.maximum(0, pos_dist - neg_dist + margin)
    
    # Average loss over the number of positive-negative pairs
    loss = total_loss / (num_positives * num_negatives)
    
    return loss

# Example usage
anchor = np.array([1.0, 2.0])
positives = [np.array([1.1, 2.1]), np.array([0.9, 1.9])]
negatives = [np.array([3.0, 3.0]), np.array([4.0, 4.0]), np.array([5.0, 5.0]), np.array([6.0, 6.0]), np.array([7.0, 7.0])]

# Calculate Triplet Loss with margin = 1.0
loss = triplet_loss(anchor, positives, negatives, margin=1.0)
print(f'Triplet Loss: {loss}')
