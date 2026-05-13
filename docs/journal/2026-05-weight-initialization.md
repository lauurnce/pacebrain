# Study notes — weight initialisation

Initialising every weight to zero does not train at all, and the reason is
worth stating precisely: every unit in a layer would compute the same thing,
receive the same gradient, and update identically. Symmetry is never broken, so
a wide layer behaves like a single unit forever.

## Why the scale matters

Signal passing forward through many layers gets multiplied by the weights each
time. Too small and the activations shrink toward zero; too large and they
blow up. The same applies in reverse for gradients. So initialisation is
really a variance-preservation problem: pick a scale that keeps the variance of
the activations roughly constant from layer to layer.

## The two standard answers

**Xavier / Glorot** — scale by `1 / sqrt(fan_in + fan_out)`. Derived assuming a
symmetric activation like tanh.

**He / Kaiming** — scale by `sqrt(2 / fan_in)`. The factor of 2 compensates for
ReLU zeroing half the inputs. This is the right default for ReLU networks.

## The practical note

Frameworks already do something sensible by default, so this is rarely a thing
to set by hand. It is worth understanding anyway, because "the loss does not
move at all" and "the loss is NaN on the first step" are both symptoms that
point here before they point anywhere else.
