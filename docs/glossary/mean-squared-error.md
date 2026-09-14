# Mean squared error

The average of the squared differences between predictions and targets: `MSE = mean((y_pred - y_true)^2)`. Squaring makes every error positive, weights large misses far more than small ones, and gives a loss whose gradient with respect to a prediction is proportional to the residual, which is smooth and easy to optimise. The price is units: MSE is in the target's units squared, so its square root (RMSE) is the number to quote when a person needs to read the error.

## See also

- [Gradient](gradient.md)
- [Huber loss](huber-loss.md)
- [Loss function](loss-function.md)
- [Residual](residual.md)

In PaceBrain, `train.py`, `train_finish.py` and `train_pacing.py` all train with `nn.MSELoss()`. `train_pacing.py` reports the square root of its best validation MSE as an RMSE in min/km, and `eval.py` scores finish-time predictions with `rmse_minutes` alongside MAE. For padded, variable-length races the plain mean would include the padding, so `seq_models.masked_mse_loss` averages over real timesteps only.
