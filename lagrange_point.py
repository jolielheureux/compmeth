"""
HW3 - Exercise 6.16: The Lagrange Point

I'm finding the L1 Lagrange point between a big body and a small body that
orbits it, like the Earth and the Moon. At L1 a satellite orbits the big body
in sync with the small body (same angular velocity omega), so it always stays
in between the two.

Where the equation comes from:
At L1 the satellite moves in a circle of radius r around the big body. The big
body pulls it inward (GM/r^2) and the small body pulls it outward
(Gm/(R - r)^2). The net inward pull has to equal the centripetal acceleration
omega^2 * r:

    GM/r^2 - Gm/(R - r)^2 = omega^2 * r

(This assumes circular orbits and that M is much bigger than m, so the big
body stays put at the center.)

This is a quintic in r, so there's no closed-form solution. I solve it with
Newton's method by rewriting it as f(r) = 0 and iterating until the step size
is smaller than a tolerance (1 m by default).

I also added L2, which is on the far side of the small body. There both bodies
pull inward, so the equation becomes:

    GM/r^2 + Gm/(r - R)^2 = omega^2 * r

I used astropy.units and astropy.constants so every number carries its
units.

I used argparse so you can pick the system from the command line.
The default is the Earth-Moon system with the values given in the problem.
Custom values can be typed with any units astropy understands.

Examples:
    python lagrange_point.py                          # Earth-Moon L1
    python lagrange_point.py --system sun-earth --point both
    python lagrange_point.py --steps
    python lagrange_point.py --M "1 solMass" --m "1 jupiterMass" --R "5.2 au"
"""
import argparse

import astropy.units as u
from astropy.constants import G, M_earth, M_sun, M_jup

# Built-in systems: (big mass, small mass, distance, omega), all with units.
# For Earth-Moon I use the omega given in the problem. For the others I set
# omega to None and calculate it from Kepler's third law instead.
# astropy.constants doesn't have the Moon's or Mars's mass, so I typed those in.
SYSTEMS = {
    "earth-moon":  (M_earth, 7.348e22 * u.kg, 3.844e8 * u.m, 2.662e-6 / u.s),
    "sun-earth":   (M_sun,   M_earth,         1.0 * u.au,    None),
    "sun-mars":    (M_sun,   6.417e23 * u.kg, 1.524 * u.au,  None),
    "sun-jupiter": (M_sun,   M_jup,           5.204 * u.au,  None),
}


def f_L1(r, M, m, R, omega):
    """
    L1 equation moved all to one side, so L1 is where this equals zero.

    r is the distance from the center of the big body.
    Big body's pull - small body's pull - centripetal acceleration.
    Every term is an acceleration, so this comes out in m / s^2.
    """
    return (G*M/r**2 - G*m/(R - r)**2 - omega**2 * r).to(u.m / u.s**2)


def f_L1_prime(r, M, m, R, omega):
    """
    Derivative of f_L1 with respect to r (Newton's method needs it).

    I took the derivative of each term by hand:
      d/dr [GM/r^2]        = -2GM/r^3
      d/dr [-Gm/(R - r)^2] = -2Gm/(R - r)^3   (chain rule gives the minus sign)
      d/dr [-omega^2 r]    = -omega^2
    Acceleration per distance, so this comes out in 1 / s^2.
    """
    return (-2*G*M/r**3 - 2*G*m/(R - r)**3 - omega**2).to(1 / u.s**2)


def f_L2(r, M, m, R, omega):
    """
    L2 equation moved all to one side. L2 is past the small body, so both
    bodies pull inward. That's why the small-body term is + and uses (r - R).
    """
    return (G*M/r**2 + G*m/(r - R)**2 - omega**2 * r).to(u.m / u.s**2)


def f_L2_prime(r, M, m, R, omega):
    """Derivative of f_L2 with respect to r, in 1 / s^2."""
    return (-2*G*M/r**3 - 2*G*m/(r - R)**3 - omega**2).to(1 / u.s**2)


def newton(func, deriv, r, args, tolerance=1 * u.m, max_steps=50, show_steps=False):
    """
    Solve func(r) = 0 with Newton's method.

    func, deriv : the function and its derivative
    r           : starting guess (a length Quantity)
    args        : extra inputs for func and deriv, here (M, m, R, omega)
    tolerance   : stop once a step is smaller than this (a length Quantity)
    max_steps   : give up after this many steps so it can't loop forever
    show_steps  : if True, print every step so I can watch it converge

    Each step moves to where the tangent line at r crosses zero:
        r_new = r - f(r) / f'(r)
    (m/s^2) / (1/s^2) = m, so the step comes out as a length automatically.
    """
    for i in range(max_steps):
        step = (func(r, *args) / deriv(r, *args)).to(u.m)
        r = r - step
        if show_steps:
            print(f"  step {i+1}: r = {r:.6e}   (step was {step:.3e})")
        if abs(step) < tolerance:
            return r
    print("Warning: Newton's method did not converge!")
    return r


def quantity(kind):
    """
    Makes an argparse 'type' that reads a number with units, like "5.2 au",
    and checks it's the right kind of unit (mass, length, ...).
    If you type just a number, I assume SI units (kg, m, 1/s, m).
    """
    si = {"mass": u.kg, "length": u.m, "frequency": 1 / u.s}[kind]

    def convert(text):
        try:
            q = u.Quantity(text)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"can't read '{text}' as a {kind}")
        if q.unit == u.dimensionless_unscaled:   # just a number, so use SI
            q = q.value * si
        if not q.unit.is_equivalent(si):
            raise argparse.ArgumentTypeError(f"'{text}' is not a {kind}")
        return q.to(si)

    return convert


def parse_args():
    """Set up the command-line options."""
    parser = argparse.ArgumentParser(
        description="Find the L1 (and L2) Lagrange points with Newton's method. "
                    "Default system is the Earth and Moon. Values can include "
                    "units, e.g. --M '1 solMass' --R '5.2 au'."
    )
    parser.add_argument("--system", choices=SYSTEMS.keys(), default="earth-moon",
                        help="which built-in system to use (default: earth-moon)")
    # These let you set up your own system. If you give any of them, they
    # replace that value from the chosen --system.
    parser.add_argument("--M", type=quantity("mass"), metavar="MASS",
                        help="big body's mass, e.g. '1 solMass' (plain number = kg)")
    parser.add_argument("--m", type=quantity("mass"), metavar="MASS",
                        help="small body's mass, e.g. '1 earthMass' (plain number = kg)")
    parser.add_argument("--R", type=quantity("length"), metavar="DIST",
                        help="distance between the bodies, e.g. '1 au' (plain number = m)")
    parser.add_argument("--omega", type=quantity("frequency"), metavar="OMEGA",
                        help="angular velocity, e.g. '2.662e-6 1/s' (default: from "
                             "Kepler's 3rd law, except Earth-Moon uses the problem's value)")
    parser.add_argument("--point", choices=["L1", "L2", "both"], default="L1",
                        help="which Lagrange point to find (default: L1)")
    parser.add_argument("--guess", type=quantity("length"), metavar="DIST",
                        help="starting guess for r (default: based on the Hill radius)")
    parser.add_argument("--tolerance", type=quantity("length"), default=1 * u.m,
                        metavar="DIST",
                        help="stop when a step is smaller than this (default: 1 m)")
    parser.add_argument("--steps", action="store_true",
                        help="print every Newton's method step")
    return parser.parse_args()


def main():
    args = parse_args()

    # Start from the chosen system, then overwrite anything given on the command line
    M, m, R, omega = SYSTEMS[args.system]
    custom = any(x is not None for x in (args.M, args.m, args.R))
    if args.M is not None:
        M = args.M
    if args.m is not None:
        m = args.m
    if args.R is not None:
        R = args.R
    if custom:
        omega = None   # a new system needs a new omega unless I give one
    if args.omega is not None:
        omega = args.omega

    # Put everything in SI so the printouts are consistent
    M, m, R = M.to(u.kg), m.to(u.kg), R.to(u.m)

    # If I still don't have omega, get it from Kepler's third law:
    # omega^2 = G(M + m) / R^3
    if omega is None:
        omega = (G * (M + m) / R**3) ** 0.5
    omega = omega.to(1 / u.s)

    # Starting guess: the Hill radius h = R * (m / 3M)^(1/3) is roughly how far
    # L1 and L2 are from the small body when M >> m. So I start at R - h for L1
    # and R + h for L2. (m / 3M is unitless, so h comes out as a length.)
    h = R * (m / (3*M)).decompose() ** (1/3)
    params = (M, m, R, omega)

    name = "custom system" if custom else args.system
    print(f"System: {name}")
    print(f"  M = {M:.4e}, m = {m:.4e}, R = {R:.4e}, omega = {omega:.4e}")

    if args.point in ("L1", "both"):
        guess = args.guess if args.guess is not None else R - h
        r_L1 = newton(f_L1, f_L1_prime, guess, params, args.tolerance,
                      show_steps=args.steps)
        print(f"L1 is r = {r_L1:.4e} = {r_L1.to(u.km):.4e} from the center "
              f"of the big body ({(R - r_L1).to(u.km):.3e} before the small body)")

    if args.point in ("L2", "both"):
        guess = args.guess if args.guess is not None else R + h
        r_L2 = newton(f_L2, f_L2_prime, guess, params, args.tolerance,
                      show_steps=args.steps)
        print(f"L2 is r = {r_L2:.4e} = {r_L2.to(u.km):.4e} from the center "
              f"of the big body ({(r_L2 - R).to(u.km):.3e} past the small body)")


if __name__ == "__main__":
    main()