# Exploding gradient

The opposite failure — repeated multiplication by factors above one drives gradients to enormous values, and a single step destroys the weights. It announces itself as a loss that suddenly becomes NaN, and [[gradient-clipping]] is the standard blunt fix.
