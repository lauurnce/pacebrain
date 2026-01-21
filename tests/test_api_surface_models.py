"""API surface checks for pacebrain.models."""

def test_models_MLP_forward_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.models')
    obj = getattr(mod, 'MLP')
    obj = getattr(obj, 'forward')
    assert callable(obj)
