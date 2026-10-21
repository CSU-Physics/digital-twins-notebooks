"""Helpers for the Module 3 lab (Digital Twins for Semiconductor Manufacturing): fitting the Deal-Grove model.

Units: thickness x in nm, time t in minutes, B in nm^2/min, B/A in nm/min, A in nm.
Deal-Grove:  x^2 + A x = B (t + tau),  with  tau = (xi^2 + A xi) / B  for a starting oxide xi.

linear_form_fit()  a straight-line fit of the rearranged model, for starting values only.
fit_deal_grove()   weighted nonlinear least squares of x(t) with SciPy's curve_fit, sigma known (absolute_sigma=True).
predict()          the predicted thickness with a confidence band for the mean and a prediction band for a new measurement.
"""
import numpy as np
from scipy.optimize import curve_fit

__version__ = "1.0.0"

Z95 = 1.96          # a 95% band is the estimate plus or minus 1.96 standard errors (the stated sigma is treated as known)


def _thickness(t, B, BA, xi):
    A = B / BA
    tau = (xi ** 2 + A * xi) / B
    return A / 2.0 * (np.sqrt(1.0 + 4.0 * B * (np.asarray(t, dtype=float) + tau) / A ** 2) - 1.0)


def linear_form_fit(t, x, xi):
    """Rearranged Deal-Grove:  t = (x^2 - xi^2) / B + (x - xi) / (B/A),  a straight line in 1/B and 1/(B/A).
    Ordinary least squares on that form treats t as the noisy quantity, but the noise is in x (an errors-in-variables
    problem), so use the result only as starting values. Returns (B, BA)."""
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    M = np.column_stack([x ** 2 - xi ** 2, x - xi])
    (u, v), *_ = np.linalg.lstsq(M, t, rcond=None)
    return float(1.0 / u), float(1.0 / v)


def fit_deal_grove(t, x, sigma, xi, p0=None):
    """Fit B and B/A to measured thickness x at times t, for a known starting oxide xi and a known measurement
    standard deviation sigma (nm). Starting values come from linear_form_fit() unless p0 = (B, BA) is given.

    Returns a dict: B, BA, A, tau (minutes), se_B, se_BA (standard errors), corr (their correlation), cov (2 x 2),
    chi2 and dof (the sum of squared residuals in units of sigma, and its degrees of freedom), residuals (nm),
    and sigma, xi, n, t_min, t_max for predict()."""
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    if p0 is None:
        p0 = linear_form_fit(t, x, xi)
    model = lambda tt, B, BA: _thickness(tt, B, BA, xi)
    p, cov = curve_fit(model, t, x, p0=p0, sigma=np.full(t.size, float(sigma)), absolute_sigma=True,
                       ftol=1e-14, xtol=1e-14, gtol=1e-14, maxfev=20000)
    B, BA = float(p[0]), float(p[1])
    res = x - model(t, B, BA)
    se = np.sqrt(np.diag(cov))
    A = B / BA
    return {"B": B, "BA": BA, "A": A, "tau": (xi ** 2 + A * xi) / B, "se_B": float(se[0]), "se_BA": float(se[1]),
            "corr": float(cov[0, 1] / (se[0] * se[1])), "cov": cov, "chi2": float(np.sum((res / sigma) ** 2)),
            "dof": int(t.size - 2), "residuals": res, "sigma": float(sigma), "xi": float(xi), "n": int(t.size),
            "t_min": float(t.min()), "t_max": float(t.max())}


def predict(fit, t_new):
    """Predicted thickness at times t_new (minutes) from a fit_deal_grove() result, with 95% bands.

    se_mean comes from the parameter covariance and the gradient of x(t) with respect to B and B/A: it is the
    uncertainty of the predicted mean thickness. se_pred adds the measurement variance, sigma^2: it is the
    uncertainty of one new measurement. Returns a dict of arrays: t, x, se_mean, se_pred, conf_lo, conf_hi,
    pred_lo, pred_hi, and inside (True where t_new lies inside the calibrated time range)."""
    t_new = np.atleast_1d(np.asarray(t_new, dtype=float))
    B, BA, xi = fit["B"], fit["BA"], fit["xi"]
    x = _thickness(t_new, B, BA, xi)
    hB, hA = 1e-6 * B, 1e-6 * BA
    gB = (_thickness(t_new, B + hB, BA, xi) - _thickness(t_new, B - hB, BA, xi)) / (2 * hB)
    gA = (_thickness(t_new, B, BA + hA, xi) - _thickness(t_new, B, BA - hA, xi)) / (2 * hA)
    G = np.column_stack([gB, gA])
    se_mean = np.sqrt(np.einsum("ij,jk,ik->i", G, fit["cov"], G))
    se_pred = np.sqrt(se_mean ** 2 + fit["sigma"] ** 2)
    return {"t": t_new, "x": x, "se_mean": se_mean, "se_pred": se_pred,
            "conf_lo": x - Z95 * se_mean, "conf_hi": x + Z95 * se_mean,
            "pred_lo": x - Z95 * se_pred, "pred_hi": x + Z95 * se_pred,
            "inside": (t_new >= fit["t_min"]) & (t_new <= fit["t_max"])}
