"""API surface checks for pacebrain.data."""

def test_data_make_sample_data_make_sample_data_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.data')
    obj = getattr(mod, 'make_sample_data')
    assert callable(obj)

def test_data_load_running_csv_load_running_csv_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.data')
    obj = getattr(mod, 'load_running_csv')
    assert callable(obj)

def test_data_make_datasets_make_datasets_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.data')
    obj = getattr(mod, 'make_datasets')
    assert callable(obj)

def test_data_make_synthetic_data_make_synthetic_data_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.data')
    obj = getattr(mod, 'make_synthetic_data')
    assert callable(obj)

def test_data_train_val_split_train_val_split_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.data')
    obj = getattr(mod, 'train_val_split')
    assert callable(obj)
