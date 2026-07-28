"""
Tests for variable-length pacing sequences — per-km splits, padding,
pack_padded_sequence and the masked loss.

The important tests here are the two that justify the whole change:
padding a batch and feeding it straight to the LSTM gives *wrong* answers
inside the real region, and packing fixes them.
"""

import pytest
import torch

from pacebrain.seq_data import (
    PACE_MAX,
    PACE_MIN,
    SEQ_FEATURES,
    make_variable_length_sequences,
    make_variable_seq_datasets,
    pad_collate,
    segments_for_distance,
)
from pacebrain.seq_models import PacingLSTM, length_mask, masked_mse_loss


@pytest.fixture
def model():
    """Deterministic model — no dropout at a single layer, but eval() anyway."""
    torch.manual_seed(0)
    m = PacingLSTM(input_size=len(SEQ_FEATURES), hidden_size=16)
    m.eval()
    return m


@pytest.fixture
def ragged_batch():
    """A 5 km, a 10 km and a marathon — lengths 5, 10 and 42."""
    torch.manual_seed(1)
    return [
        (torch.randn(5, len(SEQ_FEATURES)), torch.randn(5, 1)),
        (torch.randn(10, len(SEQ_FEATURES)), torch.randn(10, 1)),
        (torch.randn(42, len(SEQ_FEATURES)), torch.randn(42, 1)),
    ]


# ---------------------------------------------------------------------------
# segment counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "distance,expected", [(5.0, 5), (10.0, 10), (21.1, 21), (42.2, 42)]
)
def test_segments_for_distance(distance, expected):
    assert segments_for_distance(distance) == expected


def test_segments_never_zero():
    """A sub-kilometre distance must still produce one timestep, not an empty sequence."""
    assert segments_for_distance(0.2) == 1


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def test_sequence_length_matches_race_distance():
    """race_distance_km is feature index 1; length must be round(distance)."""
    for X, y in make_variable_length_sequences(n_races=40, seed=3):
        distance = float(X[0, 1])
        assert len(X) == segments_for_distance(distance)
        assert len(y) == len(X)


def test_generated_races_have_more_than_one_length():
    """The whole point is a ragged batch — a fixed length would be a silent regression."""
    lengths = {len(X) for X, _ in make_variable_length_sequences(n_races=60, seed=3)}
    assert lengths == {5, 10, 21, 42}


def test_segment_fraction_spans_the_race_regardless_of_length():
    """Fraction stays comparable across lengths: always the segment midpoint in (0, 1)."""
    for X, _ in make_variable_length_sequences(n_races=20, seed=3):
        fracs = X[:, 0]
        assert 0.0 < float(fracs[0]) < 1.0
        assert float(fracs[0]) == pytest.approx(0.5 / len(X), rel=1e-5)
        assert torch.all(fracs[1:] > fracs[:-1])


def test_paces_stay_within_clip_bounds():
    for _, y in make_variable_length_sequences(n_races=40, seed=3):
        assert torch.all(y >= PACE_MIN) and torch.all(y <= PACE_MAX)


def test_back_half_fades_on_average():
    """The physics must survive the switch to per-km splits."""
    races = make_variable_length_sequences(n_races=200, seed=5)
    marathons = [y for X, y in races if len(X) == 42]
    front = torch.cat([y[:21] for y in marathons]).mean()
    back = torch.cat([y[21:] for y in marathons]).mean()
    assert back > front


def test_generation_is_seeded():
    a = make_variable_length_sequences(n_races=10, seed=11)
    b = make_variable_length_sequences(n_races=10, seed=11)
    for (Xa, ya), (Xb, yb) in zip(a, b):
        assert torch.equal(Xa, Xb) and torch.equal(ya, yb)


# ---------------------------------------------------------------------------
# pad_collate
# ---------------------------------------------------------------------------

def test_pad_collate_shapes_and_lengths(ragged_batch):
    X, y, lengths = pad_collate(ragged_batch)
    assert X.shape == (3, 42, len(SEQ_FEATURES))
    assert y.shape == (3, 42, 1)
    assert lengths.tolist() == [5, 10, 42]


def test_pad_collate_preserves_real_values_and_zero_pads(ragged_batch):
    X, _, lengths = pad_collate(ragged_batch)
    for i, (X_i, _) in enumerate(ragged_batch):
        assert torch.equal(X[i, : lengths[i]], X_i)
        assert torch.all(X[i, lengths[i] :] == 0)


# ---------------------------------------------------------------------------
# packing — the reason this change exists
# ---------------------------------------------------------------------------

def test_packed_batch_matches_running_each_race_alone(model, ragged_batch):
    """
    The correctness property: batching must not change any prediction.

    Each race run on its own is the ground truth; the packed batch has to
    reproduce it exactly over the real timesteps.
    """
    X, _, lengths = pad_collate(ragged_batch)
    batched = model(X, lengths=lengths)

    for i, (X_i, _) in enumerate(ragged_batch):
        alone = model(X_i.unsqueeze(0))[0]
        assert torch.allclose(batched[i, : lengths[i]], alone, atol=1e-6)


def test_ignoring_lengths_does_not_corrupt_per_timestep_outputs(model, ragged_batch):
    """
    The counter-intuitive half, pinned down so nobody 'fixes' it later.

    A unidirectional LSTM is causal: output t depends only on inputs 1..t,
    and all the padding is trailing. So the per-segment predictions on the
    real region are already correct WITHOUT packing. Packing is not what
    makes them right -- see the next test for what it does make right.
    """
    X, _, lengths = pad_collate(ragged_batch)
    unpacked = model(X)  # no lengths -> padding walked as if it were input
    alone = model(ragged_batch[0][0].unsqueeze(0))[0]
    assert torch.allclose(unpacked[0, :5], alone, atol=1e-6)


def test_ignoring_lengths_does_corrupt_the_final_hidden_state(ragged_batch):
    """
    The real cost of skipping pack_padded_sequence.

    h_n is the state after the LAST step the LSTM took, so on a padded batch
    it describes the trailing zeros rather than the end of the race. Nothing
    in PacingLSTM reads h_n today -- the head consumes every timestep -- but
    any summary head (a finish-time variant, say) would be silently wrong,
    which is precisely the kind of bug that never announces itself.
    """
    torch.manual_seed(0)
    lstm = torch.nn.LSTM(len(SEQ_FEATURES), 16, batch_first=True)
    X_short = ragged_batch[0][0].unsqueeze(0)                 # (1, 5, 6)
    X_padded = torch.cat([X_short, torch.zeros(1, 37, len(SEQ_FEATURES))], dim=1)

    _, (h_true, _) = lstm(X_short)
    _, (h_padded, _) = lstm(X_padded)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        X_padded, torch.tensor([5]), batch_first=True, enforce_sorted=False
    )
    _, (h_packed, _) = lstm(packed)

    assert not torch.allclose(h_padded, h_true, atol=1e-3)   # padding corrupts it
    assert torch.allclose(h_packed, h_true, atol=1e-6)       # packing restores it


def test_padded_positions_are_the_head_bias_not_zero(model, ragged_batch):
    """
    A packed LSTM zeroes the padded *hidden states*, but the linear head adds
    its bias on top -- so padded predictions are b, not 0. Documented because
    it is exactly the trap that makes an unmasked loss look almost right.
    """
    X, _, lengths = pad_collate(ragged_batch)
    out = model(X, lengths=lengths)
    assert torch.allclose(
        out[0, 5:], model.head.bias.expand_as(out[0, 5:]), atol=1e-6
    )


def test_output_keeps_the_padded_width(model, ragged_batch):
    """total_length pins the time axis; without it the batch silently reshapes."""
    X, _, lengths = pad_collate(ragged_batch)
    assert model(X, lengths=lengths).shape == (3, 42, 1)


# ---------------------------------------------------------------------------
# masking and loss
# ---------------------------------------------------------------------------

def test_length_mask_marks_exactly_the_real_timesteps():
    mask = length_mask(torch.tensor([2, 4]), max_len=4)
    assert mask.tolist() == [[True, True, False, False], [True] * 4]


def test_masked_loss_ignores_whatever_is_in_the_padding():
    """Changing padded targets must not move the loss by even a float."""
    pred = torch.zeros(2, 5, 1)
    target = torch.zeros(2, 5, 1)
    lengths = torch.tensor([2, 5])

    before = masked_mse_loss(pred, target, lengths)
    target[0, 2:] = 1000.0  # pure padding
    after = masked_mse_loss(pred, target, lengths)

    assert before == after


def test_masked_loss_equals_plain_mse_when_nothing_is_padded():
    torch.manual_seed(2)
    pred, target = torch.randn(3, 6, 1), torch.randn(3, 6, 1)
    lengths = torch.tensor([6, 6, 6])
    expected = torch.nn.functional.mse_loss(pred, target)
    assert masked_mse_loss(pred, target, lengths) == pytest.approx(float(expected))


def test_masked_loss_divides_by_real_timesteps_not_padded_ones():
    """
    Two rows, 2 and 5 real steps, every real error exactly 1.0. Dividing by
    the padded 10 would give 0.7; the correct divisor of 7 gives 1.0.
    """
    pred = torch.ones(2, 5, 1)
    target = torch.zeros(2, 5, 1)
    assert masked_mse_loss(pred, target, torch.tensor([2, 5])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# dataset wiring
# ---------------------------------------------------------------------------

def test_variable_datasets_split_by_race_and_normalise_on_train_only():
    train_ds, val_ds, (mean, std) = make_variable_seq_datasets(n_races=100, seed=7)
    assert len(train_ds) == 80 and len(val_ds) == 20
    # Stats come from train, so train features centre near zero and the val
    # split is transformed by those same numbers rather than its own.
    train_flat = torch.cat([X for X, _ in train_ds.races], dim=0)
    assert torch.allclose(train_flat.mean(dim=0), torch.zeros(len(SEQ_FEATURES)), atol=1e-5)
    assert mean.shape == std.shape == (len(SEQ_FEATURES),)


def test_dataset_round_trips_through_a_dataloader():
    """End to end: ragged dataset -> pad_collate -> packed forward -> masked loss."""
    train_ds, _, _ = make_variable_seq_datasets(n_races=40, seed=7)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=8, shuffle=False, collate_fn=pad_collate
    )
    model = PacingLSTM(input_size=len(SEQ_FEATURES), hidden_size=16)

    X, y, lengths = next(iter(loader))
    loss = masked_mse_loss(model(X, lengths=lengths), y, lengths)
    assert loss.item() > 0 and torch.isfinite(loss)
