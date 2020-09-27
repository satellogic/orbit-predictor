from math import sqrt, copysign, isnan

from scipy.optimize.optimize import OptimizeResult, _status_message

inf = float("inf")


def minimize_scalar_bounded_alt(func, bounds, xatol=1e-5, maxiter=500, **extra):
    # Adapted from
    # https://github.com/scipy/scipy/blob/v1.5.2/scipy/optimize/optimize.py
    maxfun = maxiter
    x1, x2 = bounds
    assert x1 <= x2

    flag = 0

    sqrt_eps = sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x)
    num = 1
    fu = inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = 1
        # Check for parabolic fit
        if abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((abs(p) < abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = copysign(1, xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = copysign(1, rat) + (rat == 0)
        x = xf + si * max(abs(rat), tol1)
        fu = func(x)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    if isnan(xf) or isnan(fx) or isnan(fu):
        flag = 2

    fval = fx

    result = OptimizeResult(fun=fval, status=flag, success=(flag == 0),
                            message={0: 'Solution found.',
                                     1: 'Maximum number of function calls '
                                        'reached.',
                                     2: _status_message['nan']}.get(flag, ''),
                            x=xf, nfev=num)

    return result
