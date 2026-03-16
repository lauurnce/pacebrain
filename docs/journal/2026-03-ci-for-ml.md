# CI for machine learning

Testing shapes, ranges, determinism and pipeline wiring rather than model accuracy. Accuracy is too slow and too noisy to gate on, while a shape error or a leak is deterministic and exactly what a test should catch.
