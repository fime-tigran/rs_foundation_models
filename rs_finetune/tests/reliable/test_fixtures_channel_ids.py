"""Smoke tests for channel-ID fixtures in reliable/conftest.py.

TDD step 1: fixture smoke tests before implementing the fixtures.
"""


def test_training_channel_ids_default_is_rgb(training_channel_ids):
    assert training_channel_ids == [0, 1, 2]


def test_eval_superset_channel_ids_adds_nir(eval_superset_channel_ids):
    # RGB + B08 (NIR). B08 is the 7th band (index 6) in the 12-band order
    # [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12,VV,VH].
    assert eval_superset_channel_ids == [0, 1, 2, 6]


def test_eval_no_overlap_channel_ids_is_sar(eval_no_overlap_channel_ids):
    # VV, VH at indices 10, 11 in the 12-band order.
    assert eval_no_overlap_channel_ids == [10, 11]


def test_training_ids_disjoint_from_no_overlap(
    training_channel_ids, eval_no_overlap_channel_ids
):
    assert set(training_channel_ids).isdisjoint(set(eval_no_overlap_channel_ids))


def test_training_ids_subset_of_superset(
    training_channel_ids, eval_superset_channel_ids
):
    assert set(training_channel_ids).issubset(set(eval_superset_channel_ids))
