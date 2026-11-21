"""Host-side level bookkeeping: CDNest's do_bookkeeping pieces."""

import math

import numpy as np
import pytest

from jnesty.dfnest.levels import (create_level, enough_levels,
                                  kill_probabilities, recalculate_log_X,
                                  renormalise_visits)
from jnesty.dfnest.state import LevelBook


def _make_book(lam=10.0, thresholds=None, logX=None):
    book = LevelBook(16, np.random.default_rng(0), lam)
    if thresholds is not None:
        for i, value in enumerate(thresholds):
            if i == 0:
                book.logl[0] = value
            else:
                book.logl[i] = value
                book.n_levels = i + 1
    if logX is not None:
        book.logX[:len(logX)] = logX
    return book


class TestEnoughLevels:
    def test_needs_ten_levels_in_auto_mode(self):
        logl = np.concatenate([[-np.inf], np.linspace(-10, -1, 9)])
        assert not enough_levels(logl, 5, 0, 0.1)
        assert not enough_levels(logl, 9, 0, 0.1)

    def test_ptol_criterion(self):
        # Gaps of 0.05 over >= 10 levels: mean and max both below the
        # thresholds (0.1 and 0.11).
        logl = np.concatenate([[-np.inf], -np.arange(10, 0, -1) * 0.05])
        # threshold gaps are all 0.05 -> the last-20 window includes the
        # -inf first gap only when n <= 21; use 11 levels so the window
        # (10 gaps, truncated at k >= 1) excludes gap 1? It does not: with
        # 11 levels the window covers gaps 1..10 which includes the -inf
        # gap, so it cannot fire.
        assert not enough_levels(logl, 11, 0, 0.1)
        # 22+ levels push the -inf gap out of the 20-gap window.
        logl2 = np.concatenate([[-np.inf], -np.arange(30, 0, -1) * 0.05])
        assert enough_levels(logl2, 31, 0, 0.1)

    def test_max_gap_blocks_criterion(self):
        logl = np.concatenate([[-np.inf], -np.arange(30, 0, -1) * 0.05])
        logl[-1] = logl[-2] + 0.2  # one big gap at the top
        assert not enough_levels(logl, 31, 0, 0.1)

    def test_fixed_levels(self):
        logl = np.zeros(5)
        assert not enough_levels(logl, 4, 5, 0.1)
        assert enough_levels(logl, 5, 5, 0.1)

    def test_capacity_stops_creation(self):
        logl = np.zeros(1001)
        assert enough_levels(logl, 1000, 0, 0.1)
        assert enough_levels(logl, 1001, 0, 0.1)

    def test_one_level_auto_mode_not_enough(self):
        assert not enough_levels(np.array([-np.inf]), 1, 0, 0.1)

    def test_one_level_fixed_is_enough(self):
        assert enough_levels(np.array([-np.inf]), 1, 1, 0.1)


class TestCreateLevel:
    def test_cut_and_retention(self):
        book = _make_book()
        vals = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
        ties = np.linspace(0.1, 0.8, 8)
        # Shuffle so the sort matters.
        rng = np.random.default_rng(1)
        order = rng.permutation(8)
        created, v_out, t_out, threshold = create_level(
            book, vals[order], ties[order], math.e)
        assert created
        assert book.n_levels == 2
        # index = int((1/e) * 8) = 2 -> threshold is the 3rd largest.
        assert threshold == pytest.approx(3.0)
        assert book.logl[1] == pytest.approx(3.0)
        # Reservoir retains the top 2 records, in descending order.
        np.testing.assert_allclose(v_out, [5.0, 4.0])
        np.testing.assert_allclose(t_out, ties[order][np.argsort(
            -vals[order])][:2])

    def test_tiebreaker_orders_equal_values(self):
        book = _make_book()
        vals = np.array([3.0, 3.0, 1.0, 1.0])
        ties = np.array([0.2, 0.9, 0.5, 0.5])
        created, v_out, t_out, threshold = create_level(
            book, vals, ties, 10.0)
        # index = int(4/10) = 0 -> threshold is the (3.0, 0.9) record.
        assert created
        assert threshold == pytest.approx(3.0)
        assert book.tie[1] == pytest.approx(0.9)
        assert v_out.shape[0] == 0

    def test_capacity_warning(self):
        book = _make_book()
        book.n_levels = book.capacity
        with pytest.warns(UserWarning, match='capacity'):
            created, *_ = create_level(book, np.array([1.0]),
                                       np.array([0.5]), math.e)
        assert not created

    def test_new_level_counters_reset(self):
        book = _make_book()
        book.visits[1] = 100
        book.tries[1] = 50
        create_level(book, np.arange(20.0), np.zeros(20), math.e)
        assert book.n_levels == 2
        # Hmm: level 1 already existed; create a second level and check
        # its counters are zero.
        create_level(book, np.arange(20.0), np.zeros(20), math.e)
        assert book.visits[2] == 0
        assert book.tries[2] == 0
        assert book.accepts[2] == 0
        assert book.exceeds[2] == 0


class TestRecalculateLogX:
    def test_ratio_formula(self):
        book = _make_book(thresholds=[-np.inf, -1.0, -2.0])
        book.exceeds[:2] = [7, 3]
        book.visits[:2] = [14, 6]
        recalculate_log_X(book, 10.0, math.e)
        r_over_c = (1.0 / math.e) * 10.0
        expected1 = math.log((7 + r_over_c) / (14 + 10.0))
        expected2 = book.logX[1] + math.log((3 + r_over_c) / (6 + 10.0))
        assert book.logX[0] == 0.0
        assert book.logX[1] == pytest.approx(expected1)
        assert book.logX[2] == pytest.approx(expected2)

    def test_strictly_decreasing(self):
        rng = np.random.default_rng(2)
        book = _make_book(thresholds=[-np.inf, -3.0, -2.0, -1.0, 0.0])
        book.visits[:4] = rng.integers(0, 50, 4)
        book.exceeds[:4] = np.minimum(book.visits[:4],
                                      rng.integers(0, 50, 4))
        recalculate_log_X(book, 5.0, 2.0)
        assert np.all(np.diff(book.logX[:5]) < 0)

    def test_zero_counters_regularised(self):
        book = _make_book(thresholds=[-np.inf, -1.0])
        recalculate_log_X(book, 8.0, math.e)
        assert book.logX[1] == pytest.approx(
            math.log(((1.0 / math.e) * 8.0) / 8.0))


class TestRenormaliseVisits:
    def test_capping_with_smoothing(self):
        book = _make_book(thresholds=[-np.inf, -1.0])
        book.accepts[:2] = [80, 30]
        book.tries[:2] = [100, 40]
        book.visits[:2] = [60, 70]
        book.exceeds[:2] = [20, 35]
        renormalise_visits(book, 50.0)
        assert book.tries[0] == 50
        assert book.accepts[0] == int((80 + 1) / (100 + 1) * 50)
        assert book.visits[0] == 50
        assert book.exceeds[0] == int((20 + 1) / (60 + 1) * 50)
        # Level 1 stays under the cap... 70 >= 50?  visits[1] = 70 >= 50
        # -> renormalised as well.
        assert book.visits[1] == 50
        assert book.exceeds[1] == int((35 + 1) / (70 + 1) * 50)

    def test_small_counters_untouched(self):
        book = _make_book(thresholds=[-np.inf, -1.0])
        book.accepts[:2] = [3, 1]
        book.tries[:2] = [5, 2]
        book.visits[:2] = [4, 2]
        book.exceeds[:2] = [1, 1]
        renormalise_visits(book, 50.0)
        assert list(book.tries[:2]) == [5, 2]
        assert list(book.accepts[:2]) == [3, 1]
        assert list(book.visits[:2]) == [4, 2]

    def test_exceeds_never_exceeds_visits(self):
        book = _make_book(thresholds=[-np.inf, -1.0])
        book.accepts[0] = 99
        book.tries[0] = 100
        book.visits[0] = 100
        book.exceeds[0] = 100  # saturating the invariant
        renormalise_visits(book, 40.0)
        assert book.exceeds[0] <= book.visits[0] == 40


class TestKillProbabilities:
    def test_top_level_rarely_killed(self):
        probability = kill_probabilities(np.array([0.0]))[0]
        assert probability < 1e-4

    def test_deeply_lagging_almost_surely_killed(self):
        probability = kill_probabilities(np.array([-20.0]))[0]
        assert probability > 0.99

    def test_overflow_yields_certain_kill(self):
        with np.errstate(over='ignore'):
            probability = kill_probabilities(np.array([-1e4]))[0]
        assert probability == 1.0
