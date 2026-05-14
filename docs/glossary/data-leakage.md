# Data leakage

Information reaching the model during training that will not be available at prediction time, producing a validation score that is optimistically wrong. Its danger is that it never raises an error — the model looks better than it is, and the discrepancy only surfaces in production.
