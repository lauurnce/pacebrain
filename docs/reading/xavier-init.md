# Understanding the Difficulty of Training Deep Networks — Glorot & Bengio, 2010

Derives an initialisation scale that keeps activation and gradient variance
roughly constant across layers, assuming a symmetric activation.

The framing is what is valuable: initialisation is a variance-preservation
problem, not an arbitrary choice. Too small and signal decays through depth, too
large and it amplifies.

Reads as a direct precursor to the ReLU-specific version, which adjusts the
constant to account for half the units being zero.
