"""API surface checks for pacebrain.seq_data."""

def test_seq_data_make_sample_sequences_make_sample_sequences_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_data')
    obj = getattr(mod, 'make_sample_sequences')
    assert callable(obj)

def test_seq_data_make_seq_datasets_make_seq_datasets_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_data')
    obj = getattr(mod, 'make_seq_datasets')
    assert callable(obj)

def test_seq_data_segments_for_distance_segments_for_distance_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_data')
    obj = getattr(mod, 'segments_for_distance')
    assert callable(obj)

def test_seq_data_make_variable_length_sequences_make_variable_length_sequences_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_data')
    obj = getattr(mod, 'make_variable_length_sequences')
    assert callable(obj)

def test_seq_data_pad_collate_pad_collate_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_data')
    obj = getattr(mod, 'pad_collate')
    assert callable(obj)

def test_seq_data_make_variable_seq_datasets_make_variable_seq_datasets_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_data')
    obj = getattr(mod, 'make_variable_seq_datasets')
    assert callable(obj)
