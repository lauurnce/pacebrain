# Study notes — more of everything

More data, more parameters, more compute all help. They help predictably, and
predictably less each time.

## The shape

Error tends to fall as a power law in each of data, parameters and compute.
Power law means straight on a log-log plot and brutal in practice: each equal
fractional gain costs a constant *multiple* more resource, so halving error
again and again gets exponentially expensive.

## The practical consequence

The question is never "would more data help" — it almost always would. It is
whether the next increment is worth its cost against the alternatives, and a
[[learning-curve]] is what turns that into a measurement instead of an opinion.

## Where the curve bends

Against irreducible noise. No amount of anything gets below the noise floor, so
a model already close to it has nothing left to gain from scale, and the
remaining error is a property of the problem rather than of the model.
