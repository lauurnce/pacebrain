"""API surface checks for pacebrain.train."""

def test_train_make_batches_make_batches_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.train')
    obj = getattr(mod, 'make_batches')
    assert callable(obj)

def test_train_train_one_epoch_train_one_epoch_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.train')
    obj = getattr(mod, 'train_one_epoch')
    assert callable(obj)

def test_train_evaluate_evaluate_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.train')
    obj = getattr(mod, 'evaluate')
    assert callable(obj)

def test_train_main_main_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.train')
    obj = getattr(mod, 'main')
    assert callable(obj)
