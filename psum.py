import torch

# Create a sample tensor
P = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])  # Shape: (3, 3)

# Case 1: P.sum()
total_sum = P.sum()  

print('Tensor-Object: ', total_sum)
print('Sum of all entries of the Tensor = 1+2+3+4+5+6+7+8+9', 1+2+3+4+5+6+7+8+9)
print('Sum of all entries of the Tensor: total_sum.item()', total_sum.item())
# This sums all elements into a scalar tensor

# Case 2: P.sum(1, keepdims=True)
row_sums = P.sum(1, keepdims=True)  
# Result: tensor([[ 6],
#                [15],
#                [24]])  # Shape: (3, 1)

print('Tensor sum - now \'row sum\': P.sum(1,keepdims=True): ', row_sums)