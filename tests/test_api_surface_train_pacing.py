"""API surface checks for pacebrain.train_pacing."""

def test_train_pacing_train_one_epoch_train_one_epoch_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.train_pacing')
    obj = getattr(mod, 'train_one_epoch')
    assert callable(obj)

def test_train_pacing_evaluate_evaluate_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.train_pacing')
    obj = getattr(mod, 'evaluate')
    assert callable(obj)

def test_train_pacing_train_train_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.train_pacing')
    obj = getattr(mod, 'train')
    assert callable(obj)
