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


"""Update orchestration for VUMPS over one unit cell.

`cell_update` does one iteration of whichever scheme the options name.
Both schemes walk the cell from left to right, one position at a time,
but for different reasons.

The one-site scheme solves every position against the same set of
environments, so the positions are independent of each other and the
order they are visited in does not matter: `N` center-tensor
eigenproblems, `N` bond eigenproblems, then the gauge step that turns
the two sets of answers back into a unit cell. It walks in order only
because walking in order is what lets it carry one environment block
instead of holding `N`.

The two-site scheme has no choice. Each pair it solves is split by an
SVD, and the split replaces the index on the bond between the pair, so
the pass has to reach that bond with a left environment built on the
index the previous split left behind. That makes it a sweep in the
finite sense, and the price of being able to change the bond dimension
at all.

Environments
------------
An infinite chain has no boundary, so the environments are the fixed
points solved in `environ`, not accumulated boundary blocks. They go
stale the moment the state changes, so `run` re-solves them once per
iteration. That is the dominant cost of the algorithm, around half the
running time, and it is `O(1)` in the length of the cell: each solve
iterates a single block around the cell until it stops moving.

What costs memory is the cheap part that follows, spreading a fixed
point from bond 0 onto every other bond. A block scales as `chi^2 * w`,
and on a quasi-1D cylinder the cell is one column, so `N` can be dozens
while `chi` and `w` both grow with the circumference. The two sides are
not symmetric in what that spreading costs:

- The left blocks are *produced in the order the pass consumes them*.
  At bond `k` the next one is `step_left_env` of the current one, so a
  single running tensor is enough and none of them need storing.
- The right blocks are *produced against the order the pass consumes
  them*. They come off the recursion from bond 0 leftward, while the
  pass asks for bonds 1, 2, 3 and so on. Last built, first used, which
  is exactly the access pattern a sliding window serves.

So one side of a pass is **streamed**, arriving as the single block
currently in hand, and the other is **stored**, arriving as a whole
`Environment` that can keep most of itself on disk. Going left to right
the left side is the streamed one and the right side is the stored one.

Which side plays which role is a property of the direction, not of the
side, so every entry point here names its arguments by side, takes
either side in either shape, and turns away the shapes it cannot run.

Mirroring to a backward sweep
-----------------------------
Caching one side only is what ties these passes to one direction; the
finite algorithm caches both because it sweeps both ways. Four things
would change in a right-to-left pass, and nothing else:

1. Which side is streamed and which is stored, the two exchanging roles.
   The signatures already allow it; `_forward_pass_envs` and the opening
   of `cell_environments` are the only places that insist on the forward
   assignment.
2. Which of `step_left_env` / `step_right_env` advances the running block.
3. The order the positions are visited in, and so the bond each one reads.
4. Which end the two-site boundary rebuild starts from: the one bond the
   pass has not cut, which is bond 0 going right and bond 1 going left.

`Environment` already takes its direction as a setting rather than
guessing it, so the storage side needs nothing new.

Installing the result
---------------------
`AL`, `AR` and `C` go straight into the state through
`iMPS.set_canonical`, which takes all three as given. Deriving them from
the updated tensors instead would require the transfer operator of a state
caught mid-update to have a single dominant eigenvalue, which it need not.
The three are mutually consistent only at the fixed point; their mismatch
is the gradient this module reports.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from nicole import Tensor, einsum

from alice.network import iMPO, iMPS

from .environ import (
    Environment,
    build_right_envs,
    left_environment,
    spread_right_envs,
    step_left_env,
)
from .scheme_1s import gauge_tensors, optimize_bond, optimize_center
from .scheme_2s import discarded_weight, optimize_two_site, split_two_site

if TYPE_CHECKING:
    from .vumps import Options

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Which side a pass streams and which it stores
# ---------------------------------------------------------------------------

def _forward_pass_envs(
    env_left: Union[Tensor, Environment],
    env_right: Union[Tensor, Environment],
) -> Tuple[Tensor, Environment]:
    """Read a left and a right environment as the streamed and stored one.

    A left-to-right pass streams the left side and stores the right, so
    it wants a single block on the left and an `Environment` on the
    right. The other assignment describes a backward pass, which nothing
    implements yet; two sides in the same shape describe no pass at all.

    Parameters
    ----------
    env_left:
        Left environments, as a single block or as an `Environment`.
    env_right:
        Right environments, in either of the same two shapes.

    Returns
    -------
    env_left:
        Streamed block, the left fixed point on bond 0.
    env_right:
        Stored blocks, one per bond of the cell.

    Raises
    ------
    NotImplementedError
        If the left side is stored and the right side streamed.
    TypeError
        If both sides arrive in the same shape.
    """
    if isinstance(env_left, Tensor) and isinstance(env_right, Environment):
        return env_left, env_right
    if isinstance(env_left, Environment) and isinstance(env_right, Tensor):
        raise NotImplementedError(
            "a right-to-left pass is not implemented: stream the left side "
            "as the block on bond 0 and store the right side in an "
            "Environment over the cell"
        )
    raise TypeError(
        "a pass streams one side and stores the other, so the two "
        "environments cannot arrive in the same shape: got "
        f"{type(env_left).__name__} and {type(env_right).__name__}"
    )


# ---------------------------------------------------------------------------
# Environments over the unit cell
# ---------------------------------------------------------------------------

def cell_environments(
    imps: iMPS,
    impo: iMPO,
    *,
    tol: float = 1e-12,
    maxiter: int = 200,
    env_left: Optional[Environment] = None,
    env_right: Optional[Environment] = None,
) -> Tuple[Tensor, Environment, float]:
    """Solve the boundary fixed points and set up the environments of a pass.

    `imps` must be gauge-fixed. Both fixed points live on bond 0. The left
    one is handed back as it is, because a pass builds the rest of the left
    side as it goes; the right one is spread over every bond of the cell,
    because a pass reads those in the opposite order to the one they are
    built in. The module docstring gives the argument in full.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS.
    impo:
        Hamiltonian iMPO sharing the unit cell of `imps`.
    tol:
        Convergence tolerance forwarded to the fixed-point solver.
    maxiter:
        Maximum number of unit cells absorbed by the fixed-point solver.
    env_left:
        Storage for the left side, which a forward pass streams rather
        than stores. Taken for symmetry with `env_right` and refused.
    env_right:
        Storage to fill. `None` builds a plain in-memory one. A caller
        that runs many iterations should build one `Environment` up front
        and pass it in each time, so that its cache directory and its I/O
        thread are set up once rather than per iteration.

    Returns
    -------
    env_left:
        Left fixed point, the block on bond 0.
    env_right:
        Right blocks, one per bond. The same object that was passed in,
        when one was.
    energy:
        Energy of one unit cell, from the left fixed point.

    Raises
    ------
    NotImplementedError
        If storage was handed in for the left side, which describes a
        right-to-left pass.
    """
    # Only the stored side needs somewhere to live, since the streamed one
    # is a single block the pass carries itself.
    if env_left is not None:
        raise NotImplementedError(
            "a right-to-left pass is not implemented: a forward pass stores "
            "the right side, so its storage belongs to env_right"
        )
    if env_right is None:
        env_right = Environment(imps.N)

    left_block, energy = left_environment(imps, impo, tol=tol, maxiter=maxiter)
    build_right_envs(imps, impo, env_right, tol=tol, maxiter=maxiter)

    return left_block, env_right, energy


# ---------------------------------------------------------------------------
# One VUMPS iteration
# ---------------------------------------------------------------------------

def cell_update(
    imps: iMPS,
    impo: iMPO,
    env_left: Union[Tensor, Environment],
    env_right: Union[Tensor, Environment],
    opts: Options,
) -> Tuple[float, float]:
    """Run one VUMPS iteration over the whole unit cell, in place.

    `imps` must be gauge-fixed on entry and is left gauge-fixed on return,
    carrying the updated `AL`, `AR` and `C`.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS, updated in place.
    impo:
        Hamiltonian iMPO sharing the unit cell.
    env_left:
        Left fixed point on bond 0, from `cell_environments`: the
        streamed side, which the pass carries forward itself rather than
        being handed every block of.
    env_right:
        Right blocks on every bond, from `cell_environments`: the stored
        side, read in increasing bond order and left in no particular
        state.
    opts:
        Run options (scheme, Davidson parameters, truncation).

    Returns
    -------
    gradient:
        Largest `|AC - AL C|` over the cell: the VUMPS gradient norm, and
        the quantity `run` converges on. It measures how far the state is
        from being a stationary point of the energy, which is a sharper
        signal than the change in energy between iterations.
    discarded:
        Largest weight discarded by a split over the cell, or `0.0` for a
        scheme that never splits anything.

    Raises
    ------
    NotImplementedError
        If `opts.scheme` names a scheme this module does not implement,
        or if the environments describe a right-to-left pass.
    TypeError
        If both environments come in the same shape.
    ValueError
        If the two-site scheme is asked for a one-site unit cell.
    """
    env_left, env_right = _forward_pass_envs(env_left, env_right)

    if opts.scheme == '1s':
        return _update_one_site(imps, impo, env_left, env_right, opts), 0.0
    if opts.scheme == '2s':
        return _update_two_site(imps, impo, env_left, env_right, opts)
    raise NotImplementedError(f"cell_update: unknown scheme {opts.scheme!r}")


def _reconcile(
    centers: List[Tensor], bonds: List[Tensor],
) -> Tuple[List[Tensor], List[Tensor], float]:
    """Gauge every position of the cell and report the largest mismatch.

    Parameters
    ----------
    centers:
        One center tensor per site.
    bonds:
        One bond matrix per bond.

    Returns
    -------
    left_isos:
        Left-orthonormal tensors, one per site.
    right_isos:
        Right-orthonormal tensors, one per site.
    gradient:
        Largest `|AC - AL C|` over the cell.
    """
    N = len(centers)
    left_isos: List[Tensor] = []
    right_isos: List[Tensor] = []
    gradient = 0.0
    for i in range(N):
        left_iso, right_iso, error = gauge_tensors(
            centers[i], bonds[i], bonds[(i + 1) % N],
        )
        left_isos.append(left_iso)
        right_isos.append(right_iso)
        gradient = max(gradient, error)
    return left_isos, right_isos, gradient


def _davidson_opts(opts: Options) -> dict:
    """Collect the Davidson settings out of `opts`."""
    return {
        'max_iter': opts.davidson_max_iter,
        'tol': opts.davidson_tol,
        'max_subspace': opts.davidson_max_subspace,
    }


def _truncation(opts: Options) -> Optional[dict]:
    """Collect the truncation settings out of `opts`, or `None` if unset."""
    trunc: dict = {}
    if opts.max_bond is not None:
        trunc['nkeep'] = opts.max_bond
    if opts.trunc_thresh > 0.0:
        trunc['thresh'] = opts.trunc_thresh
    return trunc or None


def _update_one_site(
    imps: iMPS,
    impo: iMPO,
    env_streamed: Tensor,
    env_stored: Environment,
    opts: Options,
) -> float:
    """Solve every center tensor and bond matrix of the cell, then gauge.

    Every position is solved against the environments of the state as it
    stood when the pass began, which is what makes `|AC - AL C|` a
    gradient: refreshing a block mid-pass would leave the two sets of
    answers measured against different operators, and their mismatch
    would no longer vanish at a stationary point.

    Walking the cell in order does not weaken that. The pass steps its
    left block through the tensors it read at the top, and `set_canonical`
    is not reached until the end, so each block is the one the whole cell
    would have been given up front. The order the positions are solved in
    is free for the same reason: with the environments frozen, no solve
    reads another's answer.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS, updated in place.
    impo:
        Hamiltonian iMPO sharing the unit cell.
    env_streamed:
        Streamed side, the left fixed point on bond 0.
    env_stored:
        Stored side, the right block of every bond.
    opts:
        Run options.

    Returns
    -------
    float
        Largest `|AC - AL C|` over the cell.
    """
    N = imps.N
    davidson_opts = _davidson_opts(opts)
    w = len(str(N - 1))

    # Everything is read from the current gauge before anything is written,
    # since every position is solved against the same environments.
    centers = [imps.center_tensor(i) for i in range(N)]
    bonds = [imps.bond_matrix(k) for k in range(N)]
    left_isos = imps.left_isometries()

    carried = env_streamed
    for i in range(N):
        # Bond i first, then site i, so that the two blocks this step reads
        # come in increasing bond order and the window slides by one.
        energy, bonds[i], davidson_error = optimize_bond(
            bonds[i], carried, env_stored.fetch(i), davidson_opts,
        )
        logger.debug(
            "  bond %*d: E = %+.12g, davidson err = %.4e",
            w, i, energy, davidson_error,
        )

        energy, centers[i], davidson_error = optimize_center(
            centers[i], impo[i], carried, env_stored.fetch((i + 1) % N), davidson_opts,
        )
        logger.debug(
            "  site %*d: E = %+.12g, davidson err = %.4e",
            w, i, energy, davidson_error,
        )

        carried = step_left_env(carried, left_isos[i], impo[i])

    left_isos, right_isos, gradient = _reconcile(centers, bonds)
    imps.set_canonical(left_isos, right_isos, bonds)
    return gradient


def _update_two_site(
    imps: iMPS,
    impo: iMPO,
    env_streamed: Tensor,
    env_stored: Environment,
    opts: Options,
) -> Tuple[float, float]:
    """Sweep the bonds of the cell, solving and splitting a pair at each.

    Pair `k` spans sites `k` and `k+1` and is split on the bond between
    them, which is the only index the step changes. The outer bonds of the
    pair are left alone, so the next pair starts from something consistent
    and the cell still tiles at the end.

    The sweep carries its own environments. The left one is pushed forward
    through each freshly split tensor, since the bond it sits on has just
    been replaced. The right ones are still valid as long as the sweep has
    not reached them, which covers every pair but the last; that one spans
    the cell boundary and needs the right environment on bond 1, which the
    first pair of this same sweep invalidated, so it is rebuilt by carrying
    the untouched one on bond 0 backward through the cell.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS, updated in place.
    impo:
        Hamiltonian iMPO sharing the unit cell.
    env_streamed:
        Streamed side, the left fixed point on bond 0.
    env_stored:
        Stored side, the right block of every bond, rewritten in place
        at the cell boundary.
    opts:
        Run options.

    Returns
    -------
    gradient:
        Largest `|AC - AL C|` over the cell.
    discarded:
        Largest weight discarded by a split over the cell.

    Raises
    ------
    ValueError
        If the unit cell holds a single site. A pair then spans the cell
        twice over, and the split would have to replace the bond it is
        reading from.
    """
    N = imps.N
    if N < 2:
        raise ValueError(
            "the two-site scheme needs a unit cell of at least two sites, "
            "since a pair must span one bond and leave the others alone; "
            "use scheme='1s' or describe the same state with a doubled cell"
        )

    davidson_opts = _davidson_opts(opts)
    trunc = _truncation(opts)
    w = len(str(N - 1))

    centers = [imps.center_tensor(i) for i in range(N)]
    bonds = [imps.bond_matrix(k) for k in range(N)]
    left_isos = list(imps.left_isometries())

    carried = env_streamed
    discarded = 0.0
    for k in range(N):
        j = (k + 1) % N
        if j == 0:
            # The pair that spans the cell boundary. Bond 0 is the one bond
            # this sweep has not touched, so its right block is still the
            # one solved for, and carrying it back through the sites
            # updated since puts a current block on every bond behind it.
            _, right_isos, _ = _reconcile(centers, bonds)
            spread_right_envs(right_isos, impo, env_stored)

        AC2 = einsum('abr,bcs->acrs', left_isos[k], centers[j])
        energy, AC2, davidson_error = optimize_two_site(
            AC2, impo[k], impo[j], carried,
            env_stored.fetch((k + 2) % N), davidson_opts,
        )
        discarded = max(discarded, discarded_weight(AC2, trunc))

        left_iso, bond, right_iso = split_two_site(AC2, bonds[j].itags[0], trunc)
        bonds[j] = bond
        centers[k] = einsum('abr,bc->acr', left_iso, bond)
        centers[j] = einsum('ab,bcr->acr', bond, right_iso)

        logger.debug(
            "  pair %*d: E = %+.12g, davidson err = %.4e, D = %d",
            w, k, energy, davidson_error, bond.indices[0].dim,
        )

        # `left_iso` is already the left-orthonormal tensor of site k, since
        # the polar factor of `U S` is `U` and that of `S` is the identity,
        # so it can carry the left environment onto the new bond directly.
        carried = step_left_env(carried, left_iso, impo[k])
        left_isos[j], _, _ = gauge_tensors(centers[j], bonds[j], bonds[(j + 1) % N])

    left_isos, right_isos, gradient = _reconcile(centers, bonds)
    imps.set_canonical(left_isos, right_isos, bonds)
    return gradient, discarded
