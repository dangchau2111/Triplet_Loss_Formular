import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    # Caculate distance from anchor to positive
    pos_dist = np.sum((anchor - positive) ** 2)
    
    # Caculate distance from anchor to negative
    neg_dist = np.sum((anchor - negative) ** 2)
    
    # Caculate Triplet loss value
    loss = np.maximum(pos_dist - neg_dist + margin, 0)
    
    return loss

# Example
anchor = np.array([1.0, 1.0])
positive = np.array([1.1, 1.1])
negative = np.array([2.0, 2.0])

# Calculate Triplet loss with example data and margin = 1
loss = triplet_loss(anchor, positive, negative, margin=1.0)
print(f'Triplet Loss: {loss}')
