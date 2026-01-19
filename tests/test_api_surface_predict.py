"""API surface checks for pacebrain.predict."""

def test_predict_validate_inputs_validate_inputs_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.predict')
    obj = getattr(mod, 'validate_inputs')
    assert callable(obj)

def test_predict_check_training_range_check_training_range_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.predict')
    obj = getattr(mod, 'check_training_range')
    assert callable(obj)

def test_predict_format_hms_format_hms_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.predict')
    obj = getattr(mod, 'format_hms')
    assert callable(obj)

def test_predict_format_pace_format_pace_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.predict')
    obj = getattr(mod, 'format_pace')
    assert callable(obj)

def test_predict_parse_args_parse_args_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.predict')
    obj = getattr(mod, 'parse_args')
    assert callable(obj)

def test_predict_main_main_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.predict')
    obj = getattr(mod, 'main')
    assert callable(obj)
