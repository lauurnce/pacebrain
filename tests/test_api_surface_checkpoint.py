"""API surface checks for pacebrain.checkpoint."""

def test_checkpoint_detect_version_detect_version_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.checkpoint')
    obj = getattr(mod, 'detect_version')
    assert callable(obj)

def test_checkpoint_build_checkpoint_build_checkpoint_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.checkpoint')
    obj = getattr(mod, 'build_checkpoint')
    assert callable(obj)

def test_checkpoint_read_state_dict_read_state_dict_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.checkpoint')
    obj = getattr(mod, 'read_state_dict')
    assert callable(obj)

def test_checkpoint_read_scaler_read_scaler_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.checkpoint')
    obj = getattr(mod, 'read_scaler')
    assert callable(obj)

def test_checkpoint_check_feature_cols_check_feature_cols_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.checkpoint')
    obj = getattr(mod, 'check_feature_cols')
    assert callable(obj)

def test_checkpoint__require_supported__require_supported_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.checkpoint')
    obj = getattr(mod, '_require_supported')
    assert callable(obj)
