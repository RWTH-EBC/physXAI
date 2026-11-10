"""
Shape constrained least squares: a linear regression on data only expressions whose prediction is monotonic or
convex in raw data columns, guaranteed on a domain.

The model is `y = b + sum_i w_i * phi_i(x)` with terms `phi_i` (expressions of the raw columns x). Its derivative
`dy/dx_j = sum_i w_i * dphi_i/dx_j(x)` is linear in the weights, so the constraint

    s * d^k y / dx_j^k (x) >= 0   for all x in D          (k = 1 monotonicity, k = 2 convexity, s = +-1)

is a family of linear constraints, one per point of the domain D (a semi-infinite program). It is solved with the
exchange method: a quadratic program (the least squares problem) on a finite set of constraint points, the most
violated points of D are searched and added, until no violation is left.

- Domain 'box': the box of the training data (or `bounds`, e.g. the bounds of the MPC). If the derivative is
  multilinear in the columns it depends on (linear, bilinear and quadratic terms), its minimum over the box lies in
  a corner. All corners are constraint points, so the constraint holds on the whole box ('exact'). Otherwise the
  box is searched with a Sobol sample and local minimizations from its worst points ('verified').
- Domain 'data': the constraints hold on the training samples ('data').

Weights of single terms can additionally be bounded in sign (`px.Linear(monotonicities=...)`), which holds globally.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

logger = logging.getLogger("physXAI")

DOMAINS = ("box", "data")


class ShapeConstraints:
    """
    Monotonicity and convexity of the prediction with respect to raw data columns, see the module docstring.

    Args:
        monotonicities: {column: 1 (increasing) | -1 (decreasing)}.
        convexities: {column: 1 (convex) | -1 (concave)}, along the column (the second derivative with respect to
                     the column, not the joint convexity in several columns).
        domain: 'box' (default) or 'data'.
        bounds: {column: (min, max)} replacing the range of the training data in the box, e.g. the bounds of the MPC.
        tol: Tolerated violation, relative to the size of the derivative.
        max_iter: Maximum number of exchange iterations.
        n_samples: Size of the Sobol sample of the box searched for violations in every iteration.
        max_vertices: Maximum number of box corners used as constraint points (2^d for d columns of a derivative).
        ridge: Relative Tikhonov regularization of the standardized least squares problem (unique solution for
               collinear terms).
        qp_solver: CasADi QP solver ('highs', 'qpoases', 'osqp', ...).
        seed: Seed of the Sobol sample.
    """

    def __init__(
        self,
        monotonicities: dict[str, int] | None = None,
        convexities: dict[str, int] | None = None,
        domain: str = "box",
        bounds: dict[str, tuple[float, float]] | None = None,
        tol: float = 1e-6,
        max_iter: int = 30,
        n_samples: int = 4096,
        max_vertices: int = 4096,
        ridge: float = 1e-10,
        qp_solver: str = "highs",
        seed: int = 0,
    ):
        self.monotonicities = _signs(monotonicities, "monotonicities", "1 (increasing) or -1 (decreasing)")
        self.convexities = _signs(convexities, "convexities", "1 (convex) or -1 (concave)")
        if not self.monotonicities and not self.convexities:
            raise ValueError("ShapeConstraints need at least one monotonicity or convexity.")
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain '{domain}', use one of {DOMAINS}.")
        self.domain = domain
        self.bounds: dict[str, tuple[float, float]] = dict()
        for column, bound in (bounds or {}).items():
            low, high = (float(b) for b in bound)
            if not high > low:
                raise ValueError(f"bounds of '{column}' must be (min, max) with max > min, but are {bound}.")
            self.bounds[column] = (low, high)
        if tol < 0 or max_iter < 1 or n_samples < 0 or max_vertices < 1 or ridge < 0:
            raise ValueError("tol, n_samples and ridge must be >= 0, max_iter and max_vertices >= 1.")
        if not ca.has_conic(qp_solver):
            raise ValueError(f"CasADi has no QP solver '{qp_solver}'.")
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.n_samples = int(n_samples)
        self.max_vertices = int(max_vertices)
        self.ridge = float(ridge)
        self.qp_solver = qp_solver
        self.seed = int(seed)

    @property
    def columns(self) -> list[str]:
        """The constrained columns."""
        return list(dict.fromkeys([*self.monotonicities, *self.convexities]))

    def constraints(self) -> list[tuple[str, int, int]]:
        """(column, order of the derivative, sign) of every constraint."""
        return [(c, 1, s) for c, s in self.monotonicities.items()] + [(c, 2, s) for c, s in self.convexities.items()]

    def get_config(self) -> dict:
        return {
            "monotonicities": dict(self.monotonicities),
            "convexities": dict(self.convexities),
            "domain": self.domain,
            "bounds": {c: list(b) for c, b in self.bounds.items()},
            "tol": self.tol,
            "max_iter": self.max_iter,
            "n_samples": self.n_samples,
            "max_vertices": self.max_vertices,
            "ridge": self.ridge,
            "qp_solver": self.qp_solver,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config: dict) -> ShapeConstraints:
        return cls(**config)

    def __repr__(self) -> str:
        return (
            f"ShapeConstraints(monotonicities={self.monotonicities}, convexities={self.convexities}, "
            f"domain='{self.domain}')"
        )


def _signs(values: dict | None, label: str, meaning: str) -> dict[str, int]:
    values = dict(values or {})
    wrong = {c: s for c, s in values.items() if s not in (1, -1)}
    if wrong:
        raise ValueError(f"{label} must be {meaning}, but are {wrong}.")
    return {c: int(s) for c, s in values.items()}


@dataclass
class ShapeFitResult:
    """
    Result of `fit_shape_constrained`: weights in the units of the data and the diagnosis of the constraints.

    Every entry of `constraints` holds a certificate (see the module docstring): 'exact', 'verified', 'data' or
    'violated' (no convergence after `max_iter`, see `max_violation`).
    """

    weights: np.ndarray
    bias: float
    constraints: list[dict] = field(default_factory=list)
    rmse_train: float = float("nan")
    rmse_train_unconstrained: float = float("nan")
    iterations: int = 0
    seconds: float = 0.0
    domain: dict = field(default_factory=dict)

    @property
    def certified(self) -> bool:
        """Whether no constraint is violated."""
        return all(c["certificate"] != "violated" for c in self.constraints)

    def get_config(self) -> dict:
        return {
            "constraints": self.constraints,
            "rmse_train": self.rmse_train,
            "rmse_train_unconstrained": self.rmse_train_unconstrained,
            "iterations": self.iterations,
            "seconds": self.seconds,
            "domain": self.domain,
        }


class _Constraint:
    """One shape constraint: its derivative of the terms as CasADi function and the data needed by the exchange."""

    def __init__(self, column: str, order: int, sign: int, phi: ca.SX, x: ca.SX, names: list[str], scales: np.ndarray):
        self.column, self.order, self.sign = column, order, sign
        j = names.index(column)
        derivative = ca.jacobian(phi, x[j])
        if order == 2:
            derivative = ca.jacobian(derivative, x[j])
        self.derivative = ca.Function("d", [x], [derivative / ca.DM(scales)])  # per scaled weight
        self.depends = [k for k in range(len(names)) if ca.depends_on(derivative, x[k])]
        self.trivial = derivative.nnz() == 0  # e.g. the convexity of a linear model
        self.multilinear = self.trivial or all(
            ca.jacobian(ca.jacobian(derivative, x[k]), x[k]).nnz() == 0 for k in self.depends
        )
        self.label = ("monotonicity" if order == 1 else "convexity") + f" {column} ({'+' if sign > 0 else '-'})"

    def rows(self, points: np.ndarray) -> np.ndarray:
        """Constraint rows `sign * dphi/dx (point) / scale`, shape (points, terms)."""
        if len(points) == 0:
            return np.zeros((0, self.derivative.size1_out(0)))
        values = np.asarray(self.derivative.map(len(points))(points.T)).T
        return self.sign * values


def fit_shape_constrained(
    terms: list,
    values: dict[str, np.ndarray],
    y: np.ndarray,
    bias: bool = True,
    weight_signs: list[int] | None = None,
    shape: ShapeConstraints | None = None,
) -> ShapeFitResult:
    """
    Least squares fit of `y = b + sum_i w_i * terms_i` with sign bounds on the weights and shape constraints.

    Args:
        terms: Data only expressions (`physXAI.expr`).
        values: Training values of the raw columns {column: array}.
        y: Training targets.
        bias: Whether the model has a bias.
        weight_signs: Optional sign per term: 1 (w >= 0), -1 (w <= 0), 0 (free).
        shape: Optional shape constraints with respect to raw columns.

    Returns:
        ShapeFitResult: weights and bias in the units of the data and the diagnosis of the constraints.
    """
    start = time.perf_counter()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    features = np.column_stack([np.broadcast_to(np.asarray(t.evaluate(values), dtype=np.float64), (n,)) for t in terms])
    p = features.shape[1]
    F, target, mean, scales, y_mean, y_scale = _standardize(features, y, bias)
    problem = _LeastSquares(
        F,
        target,
        shape.ridge if shape is not None else 1e-10,
        _signs_per_term(weight_signs, p),
        shape.qp_solver if shape is not None else "highs",
    )

    unconstrained = problem.unconstrained()
    result = ShapeFitResult(weights=np.zeros(p), bias=0.0)
    result.rmse_train_unconstrained = float(np.sqrt(np.mean((F @ unconstrained - target) ** 2)) * y_scale)

    if shape is None:
        w, _ = problem.solve(np.zeros((0, p)))
        result.iterations = 1
    else:
        w, result = _exchange(terms, values, problem, scales, shape, result)
    result.weights = y_scale * w / scales
    result.bias = y_mean - float(result.weights @ mean)
    result.rmse_train = float(np.sqrt(np.mean((F @ w - target) ** 2)) * y_scale)
    result.seconds = time.perf_counter() - start
    return result


def constrained_least_squares(
    features: np.ndarray, y: np.ndarray, bias: bool = True, weight_signs: list[int] | None = None, ridge: float = 1e-10
) -> tuple[np.ndarray, float]:
    """
    Least squares `y = b + features @ w` with optional sign bounds on the weights, deterministic (no seed).

    Used for the closed form solvers and as initialization of gradient based training (warm start). The problem is
    standardized and solved in QR coordinates, sign bounds as QP (HiGHS).

    Args:
        features: (samples, terms) matrix.
        y: Targets.
        bias: Whether to fit a bias.
        weight_signs: Optional sign per term: 1 (w >= 0), -1 (w <= 0), 0 (free).
        ridge: Relative Tikhonov regularization (unique solution for collinear terms).

    Returns:
        (weights, bias) in the units of the data.
    """
    features = np.asarray(features, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    F, target, mean, scales, y_mean, y_scale = _standardize(features, y, bias)
    problem = _LeastSquares(F, target, ridge, _signs_per_term(weight_signs, features.shape[1]), "highs")
    w, success = problem.solve(np.zeros((0, features.shape[1])))
    if not success:
        logger.warning("The constrained least squares problem was not solved successfully.")
    weights = y_scale * w / scales
    return weights, y_mean - float(weights @ mean)


def _standardize(features: np.ndarray, y: np.ndarray, bias: bool):
    """Standardized terms and target. Centering removes the (unconstrained) bias exactly."""
    p = features.shape[1]
    mean = features.mean(axis=0) if bias else np.zeros(p)
    scales = features.std(axis=0)
    scales[scales < 1e-12] = 1.0
    y_mean = float(y.mean()) if bias else 0.0
    y_scale = float(y.std()) or 1.0
    return (features - mean) / scales, (y - y_mean) / y_scale, mean, scales, y_mean, y_scale


def _signs_per_term(weight_signs: list[int] | None, p: int) -> list[int]:
    signs = list(weight_signs) if weight_signs is not None else [0] * p
    if len(signs) != p or any(s not in (-1, 0, 1) for s in signs):
        raise ValueError(f"weight_signs needs one sign (-1, 0, 1) per term ({p}), but is {signs}.")
    return signs


class _LeastSquares:
    """
    `min (1/N) |F w - t|^2 + lambda |w|^2` subject to linear constraints `A w >= 0`, solved in the orthogonal coordinates
    `z = R w` of the QR decomposition of `[F; sqrt(N lambda) I]`: the objective becomes `|z - c|^2` (Hessian = identity),
    so the accuracy does not suffer from the squared condition number of the normal equations `F^T F`.
    """

    def __init__(self, F: np.ndarray, target: np.ndarray, ridge: float, weight_signs: list[int], solver: str):
        n, p = F.shape
        lam = ridge * max(float(np.sum(F**2)) / n / p, 1e-12)
        Q, R = np.linalg.qr(np.vstack([F, np.sqrt(n * lam) * np.eye(p)]))
        self.c = Q.T @ np.concatenate([target, np.zeros(p)])
        self.R_inv = np.linalg.solve(R, np.eye(p))
        self.sign_rows = np.array([s * np.eye(p)[i] for i, s in enumerate(weight_signs) if s != 0]).reshape(-1, p)
        self.solver = solver
        self.p = p

    def unconstrained(self) -> np.ndarray:
        return self.R_inv @ self.c

    def solve(self, rows: np.ndarray) -> tuple[np.ndarray, bool]:
        """Weights subject to the sign bounds and `rows @ w >= 0`."""
        A = np.vstack([self.sign_rows, rows]) @ self.R_inv
        if len(A) == 0:
            return self.unconstrained(), True
        A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)
        z, success = _solve_qp(np.eye(self.p), -self.c, A, self.solver)
        return self.R_inv @ z, success


def _exchange(terms, values, problem: _LeastSquares, scales, shape: ShapeConstraints, result: ShapeFitResult):
    names = list(dict.fromkeys(v for t in terms for v in t.free_vars()))
    unknown = [c for c in [*shape.columns, *shape.bounds] if c not in names]
    if unknown:
        raise ValueError(f"Shape constraints on columns {unknown}, which are no inputs of the terms {names}.")
    x = ca.SX.sym("x", len(names))
    phi = ca.vertcat(*[t.to_casadi({name: x[k] for k, name in enumerate(names)}) for t in terms])
    constraints = [_Constraint(c, k, s, phi, x, names, scales) for c, k, s in shape.constraints()]

    data = np.column_stack([np.asarray(values[name], dtype=np.float64) for name in names])
    data_range = [(float(np.nanmin(data[:, k])), float(np.nanmax(data[:, k]))) for k in range(len(names))]
    box = [shape.bounds.get(name, data_range[k]) for k, name in enumerate(names)]
    low = np.array([b[0] for b in box])
    high = np.array([b[1] if b[1] > b[0] else b[0] + 1e-9 for b in box])  # constant columns
    result.domain = {
        "type": shape.domain,
        "box": {name: [float(low[k]), float(high[k])] for k, name in enumerate(names)},
    }

    # Candidates: training samples and, for the box, a Sobol sample. Rows are computed once.
    candidates, points = list(), list()
    for c in constraints:
        rows, pts = c.rows(data), data
        if shape.domain == "box" and not c.trivial:
            sample = _sobol(low, high, shape.n_samples, shape.seed) if shape.n_samples else np.zeros((0, len(names)))
            rows, pts = np.vstack([rows, c.rows(sample)]), np.vstack([pts, sample])
        finite = np.all(np.isfinite(rows), axis=1)
        if not finite.all():
            logger.warning(
                "%s: the derivative is not finite at %d candidate points, they are left out. Restrict the "
                "box with `bounds` to the domain where the terms are defined.",
                c.label,
                (~finite).sum(),
            )
        candidates.append(rows[finite])
        points.append(pts[finite])

    # Initial constraint points: the corners of the box (the whole box for multilinear derivatives)
    active_rows = [np.zeros((0, problem.p)) for _ in constraints]
    corners_complete = [False] * len(constraints)
    for i, c in enumerate(constraints):
        if c.trivial or shape.domain != "box":
            continue
        if 2 ** len(c.depends) <= shape.max_vertices:
            corners = _corners(low, high, c.depends)
            rows = c.rows(corners)
            finite = np.all(np.isfinite(rows), axis=1)
            active_rows[i] = rows[finite]
            corners_complete[i] = bool(finite.all())

    w = None
    success = False
    worst = [0.0] * len(constraints)
    for iteration in range(1, shape.max_iter + 1):
        w, success = problem.solve(np.vstack(active_rows))
        if not success:
            logger.warning("QP solver '%s' did not report success in iteration %d.", shape.qp_solver, iteration)
        added = False
        for i, c in enumerate(constraints):
            if c.trivial:
                worst[i] = 0.0
                continue
            violation = _normalized(candidates[i], w)
            new_rows = list()
            if len(violation):
                order = np.argsort(violation)
                bad = order[violation[order] < -shape.tol][:64]
                new_rows.append(candidates[i][bad])
                worst[i] = float(min(violation.min(), 0.0))
            if shape.domain == "box" and not (c.multilinear and corners_complete[i]) and len(violation):
                starts = points[i][np.argsort(violation)[:5]]
                local = _local_minima(c, w, starts, low, high)
                if len(local):
                    rows = c.rows(local)
                    rows = rows[np.all(np.isfinite(rows), axis=1)]
                    local_violation = _normalized(rows, w)
                    if len(local_violation):
                        worst[i] = float(min(worst[i], local_violation.min(), 0.0))
                        new_rows.append(rows[local_violation < -shape.tol])
            new_rows = [r for r in new_rows if len(r)]
            if new_rows:
                active_rows[i] = np.vstack([active_rows[i], *new_rows])
                added = True
        if not added:
            break
    result.iterations = iteration
    converged = not added

    for i, c in enumerate(constraints):
        on_data = _normalized(c.rows(data), w)
        on_data = on_data[np.isfinite(on_data)]
        # final check of the solution on all constraint points (incl. the corners), not only on the candidates
        on_points = _normalized(active_rows[i], w)
        if len(on_points):
            worst[i] = float(min(worst[i], on_points.min(), 0.0))
        if c.trivial:
            certificate = "exact" if shape.domain == "box" else "data"
        elif worst[i] < -shape.tol:  # evaluated on the final solution
            certificate = "violated"
        elif shape.domain == "data":
            certificate = "data"
        elif c.multilinear and corners_complete[i] and success:
            certificate = "exact"
        else:
            certificate = "verified"  # without a successful QP solve, the corners are only checked
        active = float(np.mean(np.abs(on_data) <= max(shape.tol, 1e-9))) if len(on_data) and not c.trivial else None
        result.constraints.append(
            {
                "column": c.column,
                "type": "monotonicity" if c.order == 1 else "convexity",
                "sign": c.sign,
                "certificate": certificate,
                "max_violation": max(0.0, -float(worst[i])),
                "constraint_points": int(len(active_rows[i])),
                "active_share_data": active,  # share of the training samples where the constraint binds
            }
        )
    if not converged:
        logger.warning("Shape constraints: no convergence after %d iterations, see the certificates.", shape.max_iter)
    for c in result.constraints:
        if c["certificate"] == "violated":
            logger.warning(
                "Shape constraint %s %s (%+d) is violated by %.3g (relative).",
                c["type"],
                c["column"],
                c["sign"],
                c["max_violation"],
            )
    return w, result


def _normalized(rows: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Constraint values `row @ w / |row|` (relative violation, scale free)."""
    if len(rows) == 0:
        return np.zeros(0)
    norms = np.linalg.norm(rows, axis=1)
    values = rows @ w
    return np.where(norms > 1e-300, values / np.maximum(norms, 1e-300), 0.0)


def _solve_qp(H, g, A, solver: str) -> tuple[np.ndarray, bool]:
    """`min 0.5 z^T H z + g^T z` subject to `A z >= 0`."""
    p = H.shape[0]
    options = {"highs": {"output_flag": False}} if solver == "highs" else {}
    if solver == "qpoases":
        options = {"printLevel": "none"}
    elif solver == "osqp":
        options = {"osqp": {"verbose": False, "eps_abs": 1e-9, "eps_rel": 1e-9, "polish": True}}
    A_dm = ca.DM(A) if len(A) else ca.DM(0, p)
    qp = ca.conic("shape_qp", solver, {"h": ca.DM(H).sparsity(), "a": A_dm.sparsity()}, options)
    m = A_dm.size1()
    solution = qp(h=ca.DM(H), g=ca.DM(g), a=A_dm, lba=ca.DM.zeros(m), uba=ca.DM.inf(m))
    return np.asarray(solution["x"]).ravel(), bool(qp.stats().get("success", False))


def _sobol(low: np.ndarray, high: np.ndarray, n: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    m = int(np.ceil(np.log2(max(n, 2))))
    sample = qmc.Sobol(d=len(low), scramble=True, seed=seed).random_base2(m)[:n]
    return qmc.scale(sample, low, high)


def _corners(low: np.ndarray, high: np.ndarray, depends: list[int]) -> np.ndarray:
    """All corners of the box in the columns `depends`, the other columns at the center."""
    center = (low + high) / 2.0
    corners = list()
    for choice in itertools.product((0, 1), repeat=len(depends)):
        point = center.copy()
        for k, c in zip(depends, choice, strict=True):
            point[k] = high[k] if c else low[k]
        corners.append(point)
    return np.array(corners)


def _local_minima(c: _Constraint, w: np.ndarray, starts: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Local minima of the constraint value over the box (L-BFGS-B), started at the worst candidates."""
    from scipy.optimize import minimize

    x = ca.SX.sym("x", len(low))
    value = c.sign * ca.dot(c.derivative(x), ca.DM(w))
    f = ca.Function("h", [x], [value, ca.gradient(value, x)])
    scale = max(float(np.max(np.abs(np.asarray(f(starts[0])[0])))), 1e-12)
    bounds = list(zip(low, high, strict=True))
    found = list()
    for start in starts:

        def fun(z):
            v, grad = f(z)
            return float(v) / scale, np.asarray(grad).ravel() / scale

        res = minimize(fun, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        if np.all(np.isfinite(res.x)):
            found.append(res.x)
    return np.array(found) if found else np.zeros((0, len(low)))
