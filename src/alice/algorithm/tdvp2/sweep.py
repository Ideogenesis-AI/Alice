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
# Author of code: Madhav Menon.


"""Forward / reverse half-sweeps for the 2-site TDVP integrator.

These sweeps reuse the DMRG 2-site machinery wholesale — the `Environment` blocks
and their transfer-matrix updates (`step_left_env` / `step_right_env`), the bond
contraction `build_bulk`, and the truncating SVD splits `split_forward` /
`split_backward`. The only differences from the DMRG sweep are:

- the local update is a *real-time evolution* of the effective Hamiltonian
  (`evolve_two_site`) rather than a Davidson eigensolve, and
- after each forward 2-site step the carried one-site tensor is evolved
  *backward* in time (`evolve_one_site` with ``-tau``) — the inverse-free TDVP
  backward correction that removes the double counting of the shared bond.

A forward half-sweep advances every bond left-to-right, leaving the orthogonality
center at site ``L-1``; the reverse half-sweep mirrors it back to site ``0``. A
symmetric (Strang) TDVP step is ``forward(dt/2)`` followed by ``reverse(dt/2)``.
"""

from __future__ import annotations

from typing import Optional

from alice.network import MPS, MPO

from ..dmrg.environ import step_left_env, step_right_env
from ..dmrg.scheme_2s import build_bulk, split_backward, split_forward
from .local import evolve_one_site, evolve_two_site


def _trunc_dict(maxdim: Optional[int], cutoff: float) -> Optional[dict]:
    """Assemble the SVD truncation dict consumed by `split_forward`/`split_backward`."""
    trunc: dict = {'thresh': max(float(cutoff), 0.0)}
    if maxdim is not None:
        trunc['nkeep'] = int(maxdim)
    return trunc


def forward_sweep(
    mps: MPS,
    mpo: MPO,
    env_left,
    env_right,
    tau: float,
    *,
    maxdim: Optional[int],
    cutoff: float,
    lanczos_tol: float,
    lanczos_maxiter: int,
) -> None:
    """Left-to-right 2-site TDVP half-sweep advancing the state by time ``tau``.

    Requires ``mps.center == 0`` and every ``env_right`` block populated. After
    the call ``mps.center == mps.L - 1``.

    For each bond ``(i, i+1)``: contract the two cores, evolve forward by ``tau``,
    SVD-split (truncating to ``maxdim``/``cutoff``) leaving ``mps[i]`` left-isometric
    and the singular values carried right, advance the left environment, and — for
    every bond except the last — evolve the carried one-site tensor backward by
    ``tau`` before it is absorbed into the next bond.
    """
    L = mps.L
    trunc = _trunc_dict(maxdim, cutoff)

    for i in range(0, L - 1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i + 1)

        theta = build_bulk(mps[i], mps[i + 1])
        theta = evolve_two_site(theta, mpo[i], mpo[i + 1], E_left, E_right, tau,
                                lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter)

        itag = mps._bond_itag(i + 1)
        mps[i], carry = split_forward(theta, itag, trunc)   # mps[i] left-iso, carry = S·V
        mps._center = i + 1

        if i == L - 1 - 1:
            mps[i + 1] = carry
            continue

        # Advance the left environment with the freshly fixed left-isometric mps[i],
        # then evolve the carried bond tensor backward in time on site i+1.
        env_left[i + 1] = step_left_env(E_left, mps[i], mpo[i])
        mps[i + 1] = evolve_one_site(
            carry, mpo[i + 1], env_left[i + 1], E_right, -tau,
            lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter,
        )


def reverse_sweep(
    mps: MPS,
    mpo: MPO,
    env_left,
    env_right,
    tau: float,
    *,
    maxdim: Optional[int],
    cutoff: float,
    lanczos_tol: float,
    lanczos_maxiter: int,
) -> None:
    """Right-to-left 2-site TDVP half-sweep advancing the state by time ``tau``.

    Requires ``mps.center == mps.L - 1`` and every ``env_left`` block populated.
    After the call ``mps.center == 0``. Mirror of :func:`forward_sweep`: the SVD
    leaves ``mps[i+1]`` right-isometric and carries the singular values left, and
    the carried one-site tensor is evolved backward by ``tau`` on site ``i``.
    """
    L = mps.L
    trunc = _trunc_dict(maxdim, cutoff)

    for i in range(L - 2, -1, -1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i + 1)

        theta = build_bulk(mps[i], mps[i + 1])
        theta = evolve_two_site(theta, mpo[i], mpo[i + 1], E_left, E_right, tau,
                                lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter)

        itag = mps._bond_itag(i + 1)
        carry, mps[i + 1] = split_backward(theta, itag, trunc)  # mps[i+1] right-iso, carry = U·S
        mps._center = i

        if i == 0:
            mps[i] = carry
            continue

        # Advance the right environment with the freshly fixed right-isometric
        # mps[i+1], then evolve the carried bond tensor backward on site i.
        env_right[i] = step_right_env(E_right, mps[i + 1], mpo[i + 1])
        mps[i] = evolve_one_site(
            carry, mpo[i], E_left, env_right[i], -tau,
            lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter,
        )
