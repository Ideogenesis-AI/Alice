# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice project.
#
# Alice is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Alice is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Alice. If not, see <https://www.gnu.org/licenses/>.


"""Tests for alice.algorithm.tgtrg.sweep.

The reference is the Taylor series itself: a step of `tau` from `ρ(β)` has to
land on `ρ(β + tau)`, which `thermal_mpo` builds independently for small
enough arguments. A cooling step is compared against it both in the free
energy (`log Z`) and in the density matrix as a whole, through the normalized
Frobenius overlap.

What the comparison can resolve is bounded by the manifold, not by the step
size, so no convergence rate is checked here: at full rank the projector
splitting integrates the flow exactly for any `tau`, and under a bond limit
the deviation settles on a floor the step size cannot lower. Those two
statements are tested in place of a rate.
"""

from __future__ import annotations

import math

import pytest

from alice.algorithm.tgtrg import Options
from alice.algorithm.tgtrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
)
from alice.algorithm.tgtrg.sweep import (
    backward_sweep,
    cool_step,
    forward_sweep,
)

from .conftest import frobenius, thermal_state


def _envs(rho, mpo, opts):
    """Seed the environment pair for a run of `opts`'s scheme."""
    L = rho.L
    two_site = opts.scheme == '2s'
    env_left = Environment(L, fetch_lo=0, fetch_hi=L - 2 if two_site else L - 1)
    env_right = Environment(L, fetch_lo=1 if two_site else 0, fetch_hi=L - 1)
    env_left[0] = left_env_boundary(rho, mpo)
    build_right_envs(rho, mpo, env_right)
    return env_left, env_right


def _step(mpo, spc, beta, tau, **kwargs):
    """Cool `ρ(beta)` by `tau` and return the density matrix and diagnostics."""
    opts = Options(**kwargs)
    rho = thermal_state(mpo, spc, beta)
    env_left, env_right = _envs(rho, mpo, opts)
    err, dw = cool_step(rho, mpo, env_left, env_right, opts, tau)
    return rho, err, dw


def _overlap_error(rho_a, rho_b) -> float:
    """Return `1 - |⟨ρ_a, ρ_b⟩|` for the two normalized tensor trains."""
    num = frobenius(rho_a, rho_b)
    denom = math.sqrt(frobenius(rho_a, rho_a) * frobenius(rho_b, rho_b))
    return abs(1.0 - abs(num) / denom)


# ---------------------------------------------------------------------------
# One step against the Taylor series
# ---------------------------------------------------------------------------

class TestAgainstTheTaylorSeries:
    """A step of `tau` from ρ(β) must land on ρ(β + tau)."""

    @pytest.mark.parametrize('scheme', ['1s', '2s', '1sp'])
    def test_log_z_matches(self, spinless_fermion_L4, scheme):
        """log Z after one step agrees with the exact value for the new β."""
        mpo, spc, exact_log_z = spinless_fermion_L4
        beta, tau = 0.1, 0.01

        rho, _, _ = _step(mpo, spc, beta, tau, scheme=scheme, order=2)
        log_z, sign = rho.log_trace()
        assert sign == 1.0
        assert log_z == pytest.approx(exact_log_z(beta + tau), rel=1e-6)

    @pytest.mark.parametrize('scheme', ['1s', '2s', '1sp'])
    def test_density_matrix_matches(self, spinless_fermion_L4, scheme):
        """The cooled ρ overlaps the Taylor ρ(β + tau) to the step's accuracy."""
        mpo, spc, _ = spinless_fermion_L4
        beta, tau = 0.1, 0.01

        rho, _, _ = _step(mpo, spc, beta, tau, scheme=scheme, order=2)
        assert _overlap_error(rho, thermal_state(mpo, spc, beta + tau)) < 1e-8

    @pytest.mark.parametrize('order', [1, 2])
    @pytest.mark.parametrize('tau', [0.1, 2.0, 8.0])
    def test_full_rank_step_is_exact(self, spinless_fermion_L4, tau, order):
        """At full rank a step is exact, however large `tau` is.

        When the manifold is the whole space the tangent projector is the
        identity, and the projector splitting integrates the flow exactly —
        no step-size error at all, so the exponential grid can take
        `tau = beta` and still land on the right density matrix. On this
        chain the 2-site scheme reaches full rank on its own, so the claim is
        testable: the cooled log Z matches the analytic value to well past
        any plausible discretization error.
        """
        mpo, spc, exact_log_z = spinless_fermion_L4
        beta = 0.5

        rho, _, _ = _step(mpo, spc, beta, tau, scheme='2s', order=order)
        log_z, _ = rho.log_trace()
        assert log_z == pytest.approx(exact_log_z(beta + tau), abs=1e-8)

    def test_rank_limit_sets_an_accuracy_floor(self, spinless_fermion_L4):
        """A bond limit, not the step size, bounds the accuracy.

        Restricted to a narrow manifold the step can only track the flow as
        closely as the manifold contains it, so shrinking `tau` stops buying
        accuracy. Only a wider `max_bond` clears that floor, and the examples
        run with it unlimited for exactly that reason.
        """
        mpo, spc, exact_log_z = spinless_fermion_L4
        beta = 0.5

        errors = []
        for tau in (0.2, 0.05):
            rho = thermal_state(mpo, spc, beta)
            rho.compact({'nkeep': 4})
            rho.canonical(0)
            opts = Options(scheme='2s', max_bond=4)
            env_left, env_right = _envs(rho, mpo, opts)
            cool_step(rho, mpo, env_left, env_right, opts, tau)
            log_z, _ = rho.log_trace()
            errors.append(abs(log_z - exact_log_z(beta + tau)))

        # Four times the step size, and barely any change in the deviation.
        assert errors[0] / errors[1] < 4.0
        assert min(errors) > 1e-6


# ---------------------------------------------------------------------------
# Step mechanics
# ---------------------------------------------------------------------------

class TestCoolStep:
    """Tests for the composition and bookkeeping of a step."""

    @pytest.mark.parametrize('scheme', ['1s', '2s', '1sp'])
    def test_order_two_returns_the_center_to_site_zero(self, spinless_fermion_L4,
                                                       scheme):
        """A symmetric step is a full sweep pair, so the center comes back."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _, _ = _step(mpo, spc, 0.1, 0.05, scheme=scheme, order=2)
        assert rho.center == 0

    @pytest.mark.parametrize('scheme', ['1s', '2s', '1sp'])
    def test_order_one_leaves_the_center_at_the_far_end(self, spinless_fermion_L4,
                                                        scheme):
        """A single forward half-sweep ends where it ends."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _, _ = _step(mpo, spc, 0.1, 0.05, scheme=scheme, order=1)
        assert rho.center == rho.L - 1

    def test_magnitude_lands_in_the_log_scale(self, spinless_fermion_L4):
        """`log_scale` grows with cooling while the tensors stay O(1).

        The magnitude of ρ(β) is unbounded; only the log can hold it, and the
        internal train has to stay normalized for the next step to be
        representable at all.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        before = rho.log_scale
        env_left, env_right = _envs(rho, mpo, Options(scheme='2s'))
        cool_step(rho, mpo, env_left, env_right, Options(scheme='2s'), 1.0)

        assert rho.log_scale > before
        assert rho[rho.center].norm() == pytest.approx(1.0, rel=1e-10)

    def test_krylov_error_is_reported(self, spinless_fermion_L4):
        """The step reports a finite, small Krylov error on this chain."""
        _, err, _ = _step(*_args(spinless_fermion_L4), scheme='2s')
        assert 0.0 <= err < 1e-6

    def test_one_site_discards_nothing(self, spinless_fermion_L4):
        """The fixed-rank scheme has no truncation to report."""
        _, _, dw = _step(*_args(spinless_fermion_L4), scheme='1s')
        assert dw == 0.0

    def test_two_site_reports_a_discarded_weight(self, spinless_fermion_L4):
        """A bond limit makes the 2-site split discard, and say so."""
        mpo, spc, _ = spinless_fermion_L4
        _, _, dw = _step(mpo, spc, 0.5, 0.5, scheme='2s', max_bond=2)
        assert dw > 0.0

    def test_bond_grows_under_the_two_site_scheme(self, spinless_fermion_L4):
        """Only the 2-site scheme can add bond dimension while cooling."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        rho.compact({'nkeep': 2})
        rho.canonical(0)
        before = max(rho.bond_dims)

        opts = Options(scheme='2s')
        env_left, env_right = _envs(rho, mpo, opts)
        cool_step(rho, mpo, env_left, env_right, opts, 0.1)
        assert max(rho.bond_dims) > before

    def test_bond_is_fixed_under_the_one_site_scheme(self, spinless_fermion_L4):
        """The 1-site tangent space has fixed rank by construction."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        rho.compact({'nkeep': 2})
        rho.canonical(0)
        before = list(rho.bond_dims)

        opts = Options(scheme='1s')
        env_left, env_right = _envs(rho, mpo, opts)
        cool_step(rho, mpo, env_left, env_right, opts, 0.1)
        assert list(rho.bond_dims) == before

    def test_two_order_one_steps_match_one_order_two_step(self,
                                                          spinless_fermion_L4):
        """Alternating `order=1` half-sweeps compose into the symmetric step.

        A forward half-sweep at `tau/2` followed by a backward one at `tau/2`
        is, by definition, the `order=2` step — so the two paths must agree
        to roundoff, not merely to the integrator's accuracy.
        """
        mpo, spc, _ = spinless_fermion_L4
        tau = 0.2

        once = thermal_state(mpo, spc, 0.1)
        opts2 = Options(scheme='2s', order=2)
        el, er = _envs(once, mpo, opts2)
        cool_step(once, mpo, el, er, opts2, tau)

        twice = thermal_state(mpo, spc, 0.1)
        opts1 = Options(scheme='2s', order=1)
        el, er = _envs(twice, mpo, opts1)
        cool_step(twice, mpo, el, er, opts1, tau / 2, forward=True)
        cool_step(twice, mpo, el, er, opts1, tau / 2, forward=False)

        assert twice.log_scale == pytest.approx(once.log_scale, rel=1e-10)
        assert _overlap_error(once, twice) < 1e-12

    def test_unsupported_order_is_rejected(self, spinless_fermion_L4):
        """A step can only be composed the two ways the integrator knows.

        `Options` refuses any other order, so the guard is reached only by
        bypassing it — but it is the kind of guard that must stay.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        opts = Options(scheme='2s')
        opts.order = 3
        env_left, env_right = _envs(rho, mpo, opts)
        with pytest.raises(NotImplementedError, match='order'):
            cool_step(rho, mpo, env_left, env_right, opts, 0.1)


# ---------------------------------------------------------------------------
# Half-sweep guards
# ---------------------------------------------------------------------------

class TestSweepGuards:
    """A cooling sweep must visit every site, so the center is checked."""

    def test_forward_sweep_requires_the_center_at_site_zero(self,
                                                            spinless_fermion_L4):
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        opts = Options(scheme='2s')
        env_left, env_right = _envs(rho, mpo, opts)
        rho.canonical(rho.L - 1)
        with pytest.raises(ValueError, match='site 0'):
            forward_sweep(rho, mpo, env_left, env_right, opts, 0.1)

    def test_backward_sweep_requires_the_center_at_the_end(self,
                                                           spinless_fermion_L4):
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        opts = Options(scheme='2s')
        env_left, env_right = _envs(rho, mpo, opts)
        with pytest.raises(ValueError, match='center at site'):
            backward_sweep(rho, mpo, env_left, env_right, opts, 0.1)

    @pytest.mark.parametrize('sweep', [forward_sweep, backward_sweep])
    def test_unknown_scheme_is_rejected_by_the_dispatcher(
        self, spinless_fermion_L4, sweep
    ):
        """Every scheme the dispatcher knows is one it can run.

        `Options` refuses an unrecognized name, so this guard is reached
        only by bypassing it — but it is the kind of guard that must stay.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        opts = Options(scheme='1s')
        env_left, env_right = _envs(rho, mpo, opts)
        if sweep is backward_sweep:
            rho.canonical(rho.L - 1)
        opts.scheme = 'nonesuch'
        with pytest.raises(NotImplementedError, match='nonesuch'):
            sweep(rho, mpo, env_left, env_right, opts, 0.1)


# ---------------------------------------------------------------------------
# The 1-site-plus scheme
# ---------------------------------------------------------------------------

class TestOneSitePlus:
    """What the bond expansion buys, and where it collapses onto `'1s'`."""

    def test_bond_grows_from_a_rank_limited_state(self, spinless_fermion_L4):
        """The point of the scheme: a 1-site-sized step that adapts the rank."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        rho.compact({'nkeep': 2})
        rho.canonical(0)
        before = max(rho.bond_dims)

        opts = Options(scheme='1sp', expand_k=4)
        env_left, env_right = _envs(rho, mpo, opts)
        cool_step(rho, mpo, env_left, env_right, opts, 0.1)
        assert max(rho.bond_dims) > before

    def test_zero_expansion_reproduces_the_one_site_scheme(self,
                                                           spinless_fermion_L4):
        """`expand_k = 0` is the one setting where the two schemes coincide.

        With no complement to add, the expanded space is the space `'1s'`
        already works in, and the two sweeps become the same arithmetic — so
        they must agree to roundoff, not merely to the integrator's accuracy.
        """
        mpo, spc, _ = spinless_fermion_L4
        beta, tau = 0.1, 0.05

        plain = thermal_state(mpo, spc, beta)
        plain.compact({'nkeep': 4})
        plain.canonical(0)
        opts_1s = Options(scheme='1s')
        el, er = _envs(plain, mpo, opts_1s)
        cool_step(plain, mpo, el, er, opts_1s, tau)

        expanded = thermal_state(mpo, spc, beta)
        expanded.compact({'nkeep': 4})
        expanded.canonical(0)
        opts_1sp = Options(scheme='1sp', expand_k=0)
        el, er = _envs(expanded, mpo, opts_1sp)
        cool_step(expanded, mpo, el, er, opts_1sp, tau)

        assert list(expanded.bond_dims) == list(plain.bond_dims)
        assert expanded.log_scale == pytest.approx(plain.log_scale, rel=1e-12)
        assert _overlap_error(plain, expanded) < 1e-12

    def test_it_tracks_the_two_site_scheme(self, spinless_fermion_L4):
        """Given room to expand, `'1sp'` lands where `'2s'` lands.

        Both adapt the rank from the same state under the same propagator;
        they differ in the space each local exponential is taken in, and on
        one step from a rank-limited state that difference stays well below
        the step's own accuracy.
        """
        mpo, spc, _ = spinless_fermion_L4
        beta, tau = 0.1, 0.05

        two_site = thermal_state(mpo, spc, beta)
        two_site.compact({'nkeep': 2})
        two_site.canonical(0)
        opts = Options(scheme='2s')
        el, er = _envs(two_site, mpo, opts)
        cool_step(two_site, mpo, el, er, opts, tau)

        one_plus = thermal_state(mpo, spc, beta)
        one_plus.compact({'nkeep': 2})
        one_plus.canonical(0)
        opts = Options(scheme='1sp', expand_k=8)
        el, er = _envs(one_plus, mpo, opts)
        cool_step(one_plus, mpo, el, er, opts, tau)

        assert _overlap_error(two_site, one_plus) < 1e-8

    def test_it_beats_the_one_site_scheme_at_equal_start(self,
                                                         spinless_fermion_L4):
        """Adapting the rank is worth something, and this is where it shows."""
        mpo, spc, exact_log_z = spinless_fermion_L4
        beta, tau = 0.5, 0.5

        errors = {}
        for scheme, kwargs in (('1s', {}), ('1sp', {'expand_k': 8})):
            rho = thermal_state(mpo, spc, beta)
            rho.compact({'nkeep': 2})
            rho.canonical(0)
            opts = Options(scheme=scheme, **kwargs)
            el, er = _envs(rho, mpo, opts)
            cool_step(rho, mpo, el, er, opts, tau)
            log_z, _ = rho.log_trace()
            errors[scheme] = abs(log_z - exact_log_z(beta + tau))

        assert errors['1sp'] < errors['1s']

    def test_a_bond_limit_is_respected_and_reported(self, spinless_fermion_L4):
        """`'1sp'` truncates like `'2s'`, cap and discarded weight included."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.5)
        rho.compact({'nkeep': 2})
        rho.canonical(0)

        opts = Options(scheme='1sp', expand_k=4, max_bond=3)
        env_left, env_right = _envs(rho, mpo, opts)
        _err, dw = cool_step(rho, mpo, env_left, env_right, opts, 0.5)

        assert max(rho.bond_dims) <= 3
        assert dw > 0.0

    def test_alpha_keeps_the_search_cheap_without_breaking_it(self,
                                                              spinless_fermion_L4):
        """A truncated Θ̃ still finds directions worth keeping.

        `expand_alpha` caps the internal bond of the factorization the
        complement is read off, so it degrades which directions are found —
        never whether they are usable. Measured against the exact search,
        the cheap one lands on all but indistinguishably the same `ρ`.
        """
        mpo, spc, _ = spinless_fermion_L4
        beta, tau = 0.1, 0.05

        states = []
        for alpha in (None, 2):
            rho = thermal_state(mpo, spc, beta)
            rho.compact({'nkeep': 2})
            rho.canonical(0)
            opts = Options(scheme='1sp', expand_k=4, expand_alpha=alpha)
            el, er = _envs(rho, mpo, opts)
            cool_step(rho, mpo, el, er, opts, tau)
            states.append(rho)

        exact_search, cheap_search = states
        assert max(cheap_search.bond_dims) > 2
        assert _overlap_error(exact_search, cheap_search) < 1e-8


def _args(fixture):
    """Unpack the L=4 fixture into `_step`'s positional arguments."""
    mpo, spc, _ = fixture
    return mpo, spc, 0.1, 0.05
