# Study notes — training in fewer bits

Using 16-bit floats for most operations roughly halves memory and speeds up
matrix work substantially on hardware that supports it.

## Why it does not just work

Half precision has a much narrower range. Small gradients underflow to zero and
disappear, and the model quietly stops learning in those parameters.

## The two fixes

**Loss scaling.** Multiply the loss by a large constant before the backward
pass so gradients land inside the representable range, then divide before the
optimiser step. Dynamic versions adjust the constant when they detect overflow.

**Master weights.** Keep a 32-bit copy of the parameters for the update, since
accumulating many small updates in 16 bits loses them to rounding.

## The takeaway

Frameworks automate all of this now, so it is one context manager. Worth
understanding anyway, because when a model trains in 32-bit and diverges in
16-bit, this is the entire list of candidate causes.
