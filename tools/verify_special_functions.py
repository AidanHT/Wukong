"""Re-measure the accuracy of the f64 special functions in crates/wukong_runtime/src/special.rs.

`erfinv`, `lgamma` and `digamma` are mirrored here operation for operation in Python doubles (IEEE
binary64, round to nearest) and compared against mpmath at 120 bits over dense random sweeps. The
Rust code uses its own <= 1 ULP `exp`/`log`/`sin`/`erf`/`erfc` (crates/wukong_runtime/src/math64.rs)
where this script uses the host libm's, so the bounds printed here describe the algorithms to
within about one ULP, which is why the Rust tests allow 4 ULP at their sample points.

Usage:  python tools/verify_special_functions.py [--samples N]
Needs:  mpmath.
"""
import argparse
import math
import random
import struct

import mpmath as mp

mp.mp.prec = 120

# --- erfinv (special.rs: ERFINV_P1/P2, Giles 2011, then Halley / log-Newton polish) -------------
P1 = [2.81022636e-08, 3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
      -0.00125372503, -0.00417768164, 0.246640727, 1.50140941]
P2 = [-0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
      -0.0076224613, 0.00943887047, 1.00167406, 2.83297682]
TWO_OVER_SQRT_PI = 1.1283791670955126


def horner(c, t):
    p = c[0]
    for k in c[1:]:
        p = k + p * t
    return p


def erfinv(y):
    if y != y:
        return y
    a = abs(y)
    if a > 1.0:
        return math.nan
    if a == 1.0:
        return math.copysign(math.inf, y)
    if a < 2.0 ** -27:
        return y * 0.886226925452758
    w = -math.log((1.0 - a) * (1.0 + a))
    p = horner(P1, w - 2.5) if w < 5.0 else horner(P2, math.sqrt(w) - 3.0)
    x = p * a
    if a <= 0.5:
        for _ in range(2):
            e = math.erf(x) - a
            d = e / (TWO_OVER_SQRT_PI * math.exp(-x * x))
            x -= d / (1.0 + x * d)
    else:
        q = 1.0 - a
        for _ in range(4):
            c = math.erfc(x)
            g = math.log(c / q)
            x += g * c / (TWO_OVER_SQRT_PI * math.exp(-x * x))
    return -x if y < 0 else x


# --- lgamma (special.rs: LGAMMA_S2 / LGAMMA_MIN / LGAMMA_STIRLING) ---------------------------------
def s2_coefficients():
    """lgamma(2 + t) = sum_k c_k t^k: c_1 = 1 - gamma, c_k = (-1)^k (zeta(k) - 1)/k."""
    with mp.workprec(200):
        return [float(1 - mp.euler)] + [float((-1) ** k * (mp.zeta(k) - 1) / k)
                                        for k in range(2, 31)]


def min_coefficients(x0):
    """lgamma(x0 + t) = lgamma(x0) + sum_k psi^(k-1)(x0)/k! t^k at the double x0."""
    with mp.workprec(200):
        x = mp.mpf(x0)
        return float(mp.loggamma(x)), [float(mp.polygamma(k - 1, x) / mp.factorial(k))
                                       for k in range(1, 27)]


S2 = s2_coefficients()
X0 = 1.4616321449683622
F0, TMIN = min_coefficients(X0)
STIRLING = [1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188, -691 / 360360, 1 / 156,
            -3617 / 122400, 43867 / 244188]
W0 = 0.4189385332046727


def series(c, t):
    acc = 0.0
    for k in reversed(c):
        acc = acc * t + k
    return acc * t


def lgamma_pos(x):
    if x < 0.5:
        return series(S2, x) - math.log1p(x) - math.log(x)
    if x < 1.25:
        t = x - 1.0
        return series(S2, t) - math.log1p(t)
    if x < 1.75:
        return F0 + series(TMIN, x - X0)
    if x < 2.5:
        return series(S2, x - 2.0)
    if x < 8.0:
        n = int(math.floor(x - 1.5))
        p, y = 1.0, x
        for _ in range(n):
            y -= 1.0
            p *= y
        return math.log(p) + series(S2, y - 2.0)
    w = 1.0 / x
    w2 = w * w
    acc = STIRLING[8]
    for c in reversed(STIRLING[:8]):
        acc = acc * w2 + c
    return (x - 0.5) * (math.log(x) - 1.0) + (W0 + w * acc)


def sin_pi(x):
    n = math.floor(x * 2.0 + 0.5)
    a = math.pi * (x - n * 0.5)
    return [math.sin(a), math.cos(a), -math.sin(a), -math.cos(a)][int(n) & 3]


def lgamma(x):
    if x != x:
        return x
    if math.isinf(x):
        return math.inf
    ax = abs(x)
    if ax < 2.0 ** -70:
        return math.inf if ax == 0 else -math.log(ax)
    if x < 0:
        if x == math.floor(x):
            return math.inf
        return math.log(math.pi / abs(sin_pi(ax) * ax)) - lgamma_pos(ax)
    return lgamma_pos(x)


# --- digamma (special.rs: DIGAMMA_C) ------------------------------------------------------------
DIGAMMA_C = [1 / 12, 1 / 120, 1 / 252, 1 / 240, 1 / 132, 691 / 32760, 1 / 12]


def digamma(x):
    if x != x or x == math.inf:
        return x
    if x == -math.inf:
        return math.nan
    res = 0.0
    if x <= 0.0:
        if x == math.floor(x):
            return math.nan
        res = -math.pi / math.tan(math.pi * (x - math.floor(x + 0.5)))
        x = 1.0 - x
    while x < 12.0:
        res -= 1.0 / x
        x += 1.0
    t = 1.0 / (x * x)
    s = DIGAMMA_C[6]
    for c in reversed(DIGAMMA_C[:6]):
        s = c - t * s
    return res + (math.log(x) - 0.5 / x - s * t)


def ordered(v):
    b = struct.unpack("<q", struct.pack("<d", v))[0]
    return -(2 ** 63) - b if b < 0 else b


def ulp(a, b):
    return abs(ordered(a) - ordered(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=5000)
    n = ap.parse_args().samples
    rng = random.Random(20261008)

    pts = [rng.uniform(-1, 1) for _ in range(n)]
    pts += [1 - 10 ** -rng.uniform(1, 15.9) for _ in range(n // 4)]
    pts += [10 ** -rng.uniform(0, 300) for _ in range(n // 4)]
    pts += [1 - 2 ** -53, 0.5, math.nextafter(0.5, 1)]
    worst = max((ulp(erfinv(y), float(mp.erfinv(y))), y) for y in pts)
    print(f"erfinv            worst {worst[0]} ULP at {worst[1]!r}")

    pts = [rng.uniform(1e-12, 12.0) for _ in range(n)]
    pts += [math.exp(rng.uniform(math.log(12.0), math.log(2.5e305))) for _ in range(n // 4)]
    pts += [c + rng.uniform(-1e-3, 1e-3) for c in (0.5, 1.0, 1.25, 1.75, 2.0, 2.5, 8.0)
            for _ in range(n // 20)]
    worst = max((ulp(lgamma(x), float(mp.loggamma(x).real)), x) for x in pts if x > 0)
    print(f"lgamma  x > 0     worst {worst[0]} ULP at {worst[1]!r}")
    pts = [-rng.uniform(1e-10, 170.0) for _ in range(n)]
    worst = max((abs(lgamma(x) - float(mp.log(abs(mp.gamma(x))))), x)
                for x in pts if x != math.floor(x))
    print(f"lgamma  x < 0     worst absolute error {worst[0]:.3g} at {worst[1]!r}")

    pts = [rng.uniform(-50, 30) for _ in range(n)]
    pts += [math.exp(rng.uniform(-20, 690)) for _ in range(n // 4)]
    worst = 0.0
    for x in pts:
        if x == math.floor(x) and x <= 0:
            continue
        r = float(mp.digamma(x))
        worst = max(worst, abs(digamma(x) - r) / max(1.0, abs(r)))
    print(f"digamma           worst error / max(1, |psi|) = {worst:.3g}")


if __name__ == "__main__":
    main()
