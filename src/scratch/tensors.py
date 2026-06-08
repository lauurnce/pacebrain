"""
Tensor basics walkthrough.

Covers: creation, shapes, ops, requires_grad, .backward(), reading gradients.
"""

import torch

print("=" * 50)
print("PART 1: Tensor creation and shapes")
print("=" * 50)

# A tensor is just a multi-dimensional array that lives on a device (CPU/GPU)
# and knows how to compute gradients.

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(3)
c = torch.ones(2, 3)        # 2 rows, 3 cols
d = torch.randn(4, 4)       # random normal

print(f"a: {a}  shape={a.shape}  dtype={a.dtype}")
print(f"b: {b}  shape={b.shape}")
print(f"c:\n{c}  shape={c.shape}")
print(f"d (4x4 random):\n{d}")

print("\n" + "=" * 50)
print("PART 2: Basic operations")
print("=" * 50)

x = torch.tensor([2.0, 4.0, 6.0])
y = torch.tensor([1.0, 2.0, 3.0])

print(f"x + y = {x + y}")
print(f"x * y = {x * y}")          # element-wise
print(f"x @ y = {x @ y}")          # dot product (same as torch.dot)
print(f"x.mean() = {x.mean()}")
print(f"x.sum() = {x.sum()}")
print(f"x ** 2 = {x ** 2}")

# Reshape: view changes the shape without copying data
matrix = torch.arange(12, dtype=torch.float32)
print(f"\narange(12): {matrix}")
matrix = matrix.view(3, 4)          # 3 rows, 4 cols — same data
print(f"view(3, 4):\n{matrix}")

print("\n" + "=" * 50)
print("PART 3: requires_grad and autograd")
print("=" * 50)

# requires_grad=True tells PyTorch to track every operation on this tensor
# so it can compute the gradient (derivative) later with .backward().

w = torch.tensor(0.5, requires_grad=True)
b_param = torch.tensor(0.0, requires_grad=True)

# Simple forward pass: z = w * 3 + b
z = w * 3.0 + b_param
print(f"z = w*3 + b = {z.item():.4f}")

# .backward() computes dz/dw and dz/db and stores them in .grad
z.backward()
print(f"dz/dw = {w.grad.item():.4f}  (should be 3.0)")
print(f"dz/db = {b_param.grad.item():.4f}  (should be 1.0)")
