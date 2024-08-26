from topicmodeling.data_loading import load_data
import numpy as np

TEST_PATH = "tests/test_data/"
TEST_CATEGORIES = ["business", "entertainment", "sport", "politics", "tech"]
RANDOM_SEED = 42


def test_load_data_no_shuffle():
    articles, labels = load_data(TEST_PATH, TEST_CATEGORIES, shuffle=False)
    assert len(articles) > 0  # at least oen article
    assert len(articles) == len(labels)  # each article has a label (which we don't use)
    assert all(category in TEST_CATEGORIES for category in labels)  # all labels should be in the test categories


def test_load_data_shuffle():
    rng = np.random.default_rng(seed=RANDOM_SEED)
    articles1, labels1 = load_data(TEST_PATH, TEST_CATEGORIES, shuffle=True, rng=rng)
    articles2, labels2 = load_data(TEST_PATH, TEST_CATEGORIES, shuffle=True, rng=rng)
    assert articles1 != articles2  # consecutive use of rng produces different results
    assert labels1 != labels2


def test_load_data_empty_categories():
    articles, labels = load_data(TEST_PATH, [])
    assert len(articles) == 0
    assert len(labels) == 0
