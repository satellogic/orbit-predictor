import numpy as np
from scipy.optimize import root_scalar

from orbit_predictor.constants import R_E_KM, F_E, OMEGA_E
from orbit_predictor.utils import mean_motion, gstime_from_datetime, orbital_period


def compute_theta(theta0, n, ecc, E, T, t0, theta_dot=OMEGA_E):
    # Equation 17 and definition of theta introduced in equation 16
    t = (E - ecc * np.sin(E)) / n + T
    return theta0 + theta_dot * (t - t0)


def compute_Z(lat, theta):
    # Equation 3
    return np.array(
        [np.cos(lat) * np.cos(theta), np.cos(lat) * np.sin(theta), np.sin(lat)]
    )


def compute_G(lat, H_m, R_eq_km=R_E_KM, f=F_E):
    # Definition of G, equivalent to geodetic to geocentric transformation
    G1 = R_eq_km / np.sqrt(1 - (2 * f - f ** 2) * np.sin(lat) ** 2) + H_m / 1000
    G2 = (1 - f) ** 2 * R_eq_km / np.sqrt(
        1 - (2 * f - f ** 2) * np.sin(lat) ** 2
    ) + H_m / 1000

    return G1 * np.cos(lat) ** 2 + G2 * np.sin(lat) ** 2


def compute_PQ(inc, raan, argp):
    # Equation 11
    P = np.array(
        [
            np.cos(argp) * np.cos(raan) - np.sin(argp) * np.sin(raan) * np.cos(inc),
            np.cos(argp) * np.sin(raan) + np.sin(argp) * np.cos(raan) * np.cos(inc),
            np.sin(argp) * np.sin(inc),
        ]
    )
    Q = np.array(
        [
            -np.sin(argp) * np.cos(raan) - np.cos(argp) * np.sin(raan) * np.cos(inc),
            -np.sin(argp) * np.sin(raan) + np.cos(argp) * np.cos(raan) * np.cos(inc),
            np.cos(argp) * np.sin(inc),
        ]
    )

    return P, Q


def compute_F(
    theta_0,
    lat_rad,
    height,
    sma,
    ecc,
    inc,
    raan,
    argp,
    T,
    E,
    *,
    t0=0.0,
    r_e=R_E_KM,
    f=F_E,
    theta_dot=OMEGA_E
):
    # Equations 8-16
    n = mean_motion(sma)
    theta = compute_theta(theta_0, n, ecc, E, T, t0, theta_dot)

    P, Q = compute_PQ(inc, raan, argp)
    Z = compute_Z(lat_rad, theta)
    G = compute_G(lat_rad, height, r_e, f)

    return (
        sma * (np.cos(E) - ecc) * (P @ Z)
        + sma * np.sqrt(1 - ecc ** 2) * np.sin(E) * (Q @ Z)
        - G
    )


def compute_Fp(
    theta_0,
    lat_rad,
    height,
    sma,
    ecc,
    inc,
    raan,
    argp,
    T,
    E,
    *,
    r_e=R_E_KM,
    f=F_E,
    theta_dot=OMEGA_E
):
    # Equations 8-16
    n = mean_motion(sma)
    theta = compute_theta(theta_0, n, ecc, E, T, 0.0)

    P, Q = compute_PQ(inc, raan, argp)
    Z = compute_Z(lat_rad, theta)

    return (
        (
            sma * (np.cos(E) - ecc) * (P[1] * Z[0] - P[0] * Z[1])
            + sma * np.sqrt(1 - ecc ** 2) * np.sin(E) * (Q[1] * Z[0] - Q[0] * Z[1])
        )
        * (1 - ecc * np.cos(E))
        * (theta_dot / n)
        + (Q @ Z) * sma * np.sqrt(1 - ecc ** 2) * np.cos(E)
        - (P @ Z) * sma * np.sin(E)
    )


def solve_E(
    theta_0,
    lat_rad,
    height,
    sma,
    ecc,
    inc,
    raan,
    argp,
    T,
    E0,
    r_e=R_E_KM,
    f=F_E,
    theta_dot=OMEGA_E
):
    res = root_scalar(
        f=lambda E: compute_F(
            theta_0,
            lat_rad,
            height,
            sma,
            ecc,
            inc,
            raan,
            argp,
            T,
            E,
            r_e=r_e,
            f=f,
            theta_dot=theta_dot,
        ),
        fprime=lambda E: compute_Fp(
            theta_0,
            lat_rad,
            height,
            sma,
            ecc,
            inc,
            raan,
            argp,
            T,
            E,
            r_e=r_e,
            f=f,
            theta_dot=theta_dot,
        ),
        x0=E0,
        method="newton",
    )
    return res.root


def example():
    import pandas as pd
    import matplotlib.pyplot as plt
    import datetime as dt

    from orbit_predictor.angles import M_to_E, E_to_M

    epoch = dt.datetime(1962, 6, 21, 0, 0, 0)

    λ = np.radians(291.5)
    φ = np.radians(76.5)
    h = np.radians(3.0)
    H_m = 900 * 0.3048

    θ_0 = gstime_from_datetime(epoch) + λ

    sma = 1.0346400 * R_E_KM
    ecc = 0.0004648
    inc = np.radians(91.995912)
    Ω = np.radians(311.45905)
    ω = np.radians(128.31922)
    T = 1371.0420 * 60

    n = mean_motion(sma)

    F_values = {}
    for t in np.arange(0, T + 20 * orbital_period(n), 10):
        E = M_to_E(n * (t - T), ecc)
        F_values[t] = compute_F(θ_0, φ, H_m, sma, ecc, inc, Ω, ω, T, E)

    ser = pd.Series(F_values)
    ax = ser.plot(color="k")
    ax.axhline(0, color="0.5", linestyle="--")

    ax.axvline(24 * 60 * 60 + 9.84 * 60, color="r")
    ax.axvline(24 * 60 * 60 + 13.13 * 60, color="r")
    ax.axvline(24 * 60 * 60 + 4 * 60 * 60 + 31.96 * 60, color="g")
    ax.axvline(24 * 60 * 60 + 4 * 60 * 60 + 37.21 * 60, color="g")
    ax.axvline(24 * 60 * 60 + 19 * 60 * 60 + 25.80 * 60, color="b")
    ax.axvline(24 * 60 * 60 + 19 * 60 * 60 + 32.00 * 60, color="b")

    # Try exact solutions
    E_sol = solve_E(θ_0, φ, H_m, sma, ecc, inc, Ω, ω, T, np.pi)
    ax.axvline(E_to_M(E_sol, ecc) / n + T, color="#ffcc00")

    plt.show()


if __name__ == "__main__":
    example()
