"""
Manual gradient descent with autograd.

Task: learn w and b so that w*x + b fits y = 2x + 1.
No nn.Module, no optimizer — just .backward() and manual parameter updates.

Key concept: after loss.backward(), each parameter's .grad holds d(loss)/d(param).
We nudge the parameter opposite to the gradient (gradient descent).
"""

import torch
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on any machine
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ----- Data ---------------------------------------------------------------
# True relationship: y = 2x + 1
# We generate noisy observations and let gradient descent recover w=2, b=1.
X = torch.linspace(-1, 1, 100)
y_true = 2.0 * X + 1.0
y_noisy = y_true + torch.randn_like(X) * 0.1

# ----- Parameters ---------------------------------------------------------
# Start with random guesses. requires_grad=True so autograd tracks them.
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# ----- Hyperparameters ----------------------------------------------------
lr = 0.1
epochs = 200

losses = []

# ----- Training loop ------------------------------------------------------
for epoch in range(epochs):
    # Forward pass: compute predictions
    y_pred = w * X + b

    # Loss: mean squared error
    loss = ((y_pred - y_noisy) ** 2).mean()
    losses.append(loss.item())

    # Backward pass: compute gradients d(loss)/dw and d(loss)/db
    loss.backward()

    # Manual parameter update — step opposite the gradient
    # torch.no_grad() prevents this update itself from being tracked
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # Zero gradients before next iteration — grads accumulate by default
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}  loss={loss.item():.6f}  "
              f"w={w.item():.4f}  b={b.item():.4f}")

print(f"\nFinal:  w={w.item():.4f} (target 2.0)  b={b.item():.4f} (target 1.0)")

# ----- Plot ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(losses)
axes[0].set_title("Loss over epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE loss")

axes[1].scatter(X.numpy(), y_noisy.numpy(), s=8, alpha=0.5, label="Noisy data")
axes[1].plot(X.numpy(), y_true.numpy(), "g--", label="True y=2x+1")
with torch.no_grad():
    y_fit = (w * X + b).numpy()
axes[1].plot(X.numpy(), y_fit, "r-", label=f"Learned y={w.item():.2f}x+{b.item():.2f}")
axes[1].set_title("Data vs learned line")
axes[1].legend()

plt.tight_layout()
plt.savefig("reports/manual_grad_descent.png", dpi=120)
print("Plot saved to reports/manual_grad_descent.png")
