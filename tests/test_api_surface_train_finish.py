"""API surface checks for pacebrain.train_finish."""

def test_train_finish_train_one_epoch_train_one_epoch_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.train_finish')
    obj = getattr(mod, 'train_one_epoch')
    assert callable(obj)

def test_train_finish_evaluate_evaluate_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.train_finish')
    obj = getattr(mod, 'evaluate')
    assert callable(obj)

def test_train_finish_train_train_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.train_finish')
    obj = getattr(mod, 'train')
    assert callable(obj)
