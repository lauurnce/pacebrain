# Study notes — reading learning curves

The most information available for the least effort. Plot training and
validation loss per epoch and most problems become visible.

## The shapes

**Both high, close together, flat.** Underfitting. The model cannot represent
the pattern. Add capacity, train longer, or improve features. Regularising
harder makes it worse.

**Training low, validation much higher and rising.** Overfitting. The gap is
the diagnosis. More data, more regularisation, or less capacity.

**Both falling and still falling at the end.** Undertrained. Just keep going.

**Training loss spiking or diverging.** Learning rate too high. Reduce it by an
order of magnitude before changing anything else.

**Validation noisy, jumping around.** Validation set too small, or batch size
too small. The signal is real but buried.

**Validation *below* training loss.** Usually not a miracle. Dropout is active
during training and off during evaluation, so the training number is measured
on a handicapped model. Regularisation makes this normal, not suspicious.

## The gap is the signal

Neither curve alone says much. Training loss near zero is not good news by
itself — a lookup table achieves it. What matters is the distance between the
two and whether it is widening.

## Early stopping reads directly off this

The epoch where validation bottoms out is the model you want. Everything after
is memorisation. Which is the entire argument for saving the best checkpoint
rather than the final one.

## Practical notes

- Log scale on the y-axis when early epochs dwarf later ones, otherwise the
  interesting part is a flat line at the bottom.
- Plot from epoch 1, including the ugly start. The first few epochs say whether
  the learning rate is sane.
- Same axes for both curves. Separate plots make the gap impossible to judge,
  and the gap is the point.
