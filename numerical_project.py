"""
============================================================
  NUMERICAL METHODS PROJECT
  Covers: Root Finding, Systems of Equations,
          Interpolation, and Numerical Integration
============================================================
"""

import math


# ──────────────────────────────────────────────
#  UTILITY HELPERS
# ──────────────────────────────────────────────

def safe_eval(expr: str, x: float) -> float:
    """Evaluate a math expression string for a given x value."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["x"] = x
    return eval(expr, {"__builtins__": {}}, allowed)


def get_float(prompt: str) -> float:
    """Prompt the user until a valid float is entered."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  ✗ Please enter a valid number.")


def get_int(prompt: str, min_val: int = 1) -> int:
    """Prompt the user until a valid integer >= min_val is entered."""
    while True:
        try:
            val = int(input(prompt))
            if val < min_val:
                print(f"  ✗ Value must be at least {min_val}.")
            else:
                return val
        except ValueError:
            print("  ✗ Please enter a valid integer.")


def get_function(prompt: str) -> str:
    """Prompt the user for a math expression and verify it parses."""
    while True:
        expr = input(prompt).strip()
        try:
            safe_eval(expr, 1.0)   # test with x = 1
            return expr
        except Exception as e:
            print(f"  ✗ Invalid expression: {e}")


def section(title: str) -> None:
    width = 56
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def divider() -> None:
    print("-" * 56)


# ──────────────────────────────────────────────
#  CHAPTER 1 – BISECTION METHOD
# ──────────────────────────────────────────────

def bisection_method() -> None:
    """
    Bisection Method – Root Finding
    Repeatedly halves an interval [a, b] where f(a)*f(b) < 0
    until the root is isolated within the given tolerance.
    """
    section("BISECTION METHOD  (Root Finding)")

    expr = get_function("  Enter f(x)  e.g. x**3 - x - 2 : ")
    a    = get_float("  Lower bound a : ")
    b    = get_float("  Upper bound b : ")

    # Validate sign change
    fa, fb = safe_eval(expr, a), safe_eval(expr, b)
    if fa * fb >= 0:
        print("\n  ✗ f(a) * f(b) must be < 0  (no guaranteed root in [a, b]).")
        print(f"    f({a}) = {fa:.6f},  f({b}) = {fb:.6f}")
        return

    tol      = get_float("  Tolerance (e.g. 0.0001) : ")
    max_iter = get_int  ("  Max iterations           : ")

    print(f"\n{'Iter':>4}  {'a':>12}  {'b':>12}  {'c (midpoint)':>14}  {'f(c)':>12}")
    divider()

    root, iterations = None, 0

    for i in range(1, max_iter + 1):
        c  = (a + b) / 2
        fc = safe_eval(expr, c)
        print(f"{i:>4}  {a:>12.6f}  {b:>12.6f}  {c:>14.6f}  {fc:>12.6f}")

        iterations = i

        if abs(fc) < tol or (b - a) / 2 < tol:
            root = c
            break

        if safe_eval(expr, a) * fc < 0:
            b = c
        else:
            a = c
    else:
        root = (a + b) / 2

    divider()
    print(f"\n  Root ≈ {root:.8f}")
    print(f"  Iterations used : {iterations}")


# ──────────────────────────────────────────────
#  CHAPTER 2 – JACOBI METHOD
# ──────────────────────────────────────────────

def _read_system(n: int):
    """Read an n×n coefficient matrix and constants vector from the user."""
    print(f"\n  Enter the coefficient matrix A  ({n}×{n}) row by row:")
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(get_float(f"    A[{i+1}][{j+1}] : "))
        A.append(row)

    print("\n  Enter the constants vector b:")
    b = [get_float(f"    b[{i+1}] : ") for i in range(n)]
    return A, b

def is_diagonally_dominant(A):
    """
    Check if matrix A is diagonally dominant.
    For every row:
        |a_ii| >= sum(|a_ij|), j != i
    """
    n = len(A)

    for i in range(n):
        diagonal = abs(A[i][i])
        others = sum(abs(A[i][j]) for j in range(n) if j != i)

        if diagonal < others:
            return False

    return True


def try_make_diagonally_dominant(A, b):
    """
    Attempt to rearrange rows to make the matrix
    diagonally dominant.

    Returns:
        (new_A, new_b, success)
    """

    n = len(A)

    used_rows = set()
    new_A = [None] * n
    new_b = [None] * n

    for col in range(n):
        found = False

        for row in range(n):

            if row in used_rows:
                continue

            diagonal = abs(A[row][col])
            others = sum(abs(A[row][j]) for j in range(n) if j != col)

            if diagonal >= others:
                new_A[col] = A[row]
                new_b[col] = b[row]
                used_rows.add(row)
                found = True
                break

        if not found:
            return A, b, False

    return new_A, new_b, True


def get_positive_float(prompt: str) -> float:
    """
    Get a positive float from the user.
    """
    while True:
        value = get_float(prompt)

        if value > 0:
            return value

        print("  ✗ Value must be greater than 0.")


def jacobi_method() -> None:
    """
    Jacobi Method – Iterative solver for linear systems Ax = b.
    Each unknown is updated using only values from the *previous* iteration.
    """
    section("JACOBI METHOD  (System of Equations)")

    n        = get_int("  Number of equations (n) : ")
    A, b     = _read_system(n)

        # Check diagonal dominance
    if not is_diagonally_dominant(A):

        print("\n  Matrix is NOT diagonally dominant.")
        print("  Attempting to rearrange rows...")

        A, b, success = try_make_diagonally_dominant(A, b)

        if success:
            print("  ✓ Successfully rearranged matrix into diagonally dominant form.")
        else:
            print("  ✗ Cannot rearrange matrix into diagonally dominant form.")
            print("    Jacobi method may not converge.")
            return

    print("\n  Enter initial guesses:")
    x = [get_float(f"    x[{i+1}]₀ : ") for i in range(n)]

    tol = get_positive_float("  Tolerance (e.g. 0.0001) : ")
    max_iter = get_int  ("  Max iterations           : ")

    header = "Iter  " + "  ".join(f"{'x'+str(i+1):>12}" for i in range(n))
    print(f"\n{header}")
    divider()

    iterations = 0
    for it in range(1, max_iter + 1):
        x_new = []
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            if A[i][i] == 0:
                print("  ✗ Zero diagonal element – Jacobi cannot proceed.")
                return
            x_new.append((b[i] - sigma) / A[i][i])

        row = f"{it:>4}  " + "  ".join(f"{v:>12.6f}" for v in x_new)
        print(row)
        iterations = it

        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            x = x_new
            break
        x = x_new

    divider()
    print("\n  Solution vector x:")
    for i, v in enumerate(x):
        print(f"    x[{i+1}] = {v:.8f}")
    print(f"  Iterations used : {iterations}")


# ──────────────────────────────────────────────
#  CHAPTER 2 – GAUSS-SEIDEL METHOD
# ──────────────────────────────────────────────

def gauss_seidel_method() -> None:
    """
    Gauss-Seidel Method – Iterative solver for linear systems Ax = b.
    Each unknown is updated immediately using the *latest* available values.
    Typically converges faster than Jacobi.
    """
    section("GAUSS-SEIDEL METHOD  (System of Equations)")

    n        = get_int("  Number of equations (n) : ")
    A, b     = _read_system(n)

        # Check diagonal dominance
    if not is_diagonally_dominant(A):

        print("\n  Matrix is NOT diagonally dominant.")
        print("  Attempting to rearrange rows...")

        A, b, success = try_make_diagonally_dominant(A, b)

        if success:
            print("  ✓ Successfully rearranged matrix into diagonally dominant form.")
        else:
            print("  ✗ Cannot rearrange matrix into diagonally dominant form.")
            print("    Gauss-Seidel method may not converge.")
            return

    print("\n  Enter initial guesses:")
    x = [get_float(f"    x[{i+1}]₀ : ") for i in range(n)]

    tol = get_positive_float("  Tolerance (e.g. 0.0001) : ")
    max_iter = get_int  ("  Max iterations           : ")

    header = "Iter  " + "  ".join(f"{'x'+str(i+1):>12}" for i in range(n))
    print(f"\n{header}")
    divider()

    iterations = 0
    for it in range(1, max_iter + 1):
        x_old = x[:]
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            if A[i][i] == 0:
                print("  ✗ Zero diagonal element – Gauss-Seidel cannot proceed.")
                return
            x[i] = (b[i] - sigma) / A[i][i]

        row = f"{it:>4}  " + "  ".join(f"{v:>12.6f}" for v in x)
        print(row)
        iterations = it

        if max(abs(x[i] - x_old[i]) for i in range(n)) < tol:
            break

    divider()
    print("\n  Solution vector x:")
    for i, v in enumerate(x):
        print(f"    x[{i+1}] = {v:.8f}")
    print(f"  Iterations used : {iterations}")


# ──────────────────────────────────────────────
#  CHAPTER 3 – LAGRANGE INTERPOLATION
# ──────────────────────────────────────────────

def lagrange_interpolation() -> None:
    """
    Lagrange Interpolation – Fits a polynomial through n data points
    and evaluates it at a target x value.
    """
    section("LAGRANGE INTERPOLATION")

    n = get_int("  Number of data points : ", min_val=2)

    print("\n  Enter x values:")
    x_pts = [get_float(f"    x[{i+1}] : ") for i in range(n)]

    print("\n  Enter y values (f(x) at each point):")
    y_pts = [get_float(f"    y[{i+1}] : ") for i in range(n)]

    # Check for duplicate x values
    if len(set(x_pts)) != len(x_pts):
        print("  ✗ x values must all be distinct.")
        return

    target = get_float("\n  Value to interpolate at x = ")

    result = 0.0
    print(f"\n{'Term':>4}  {'L_i(x)':>16}  {'y_i':>12}  {'Contribution':>16}")
    divider()

    for i in range(n):
        # Build the Lagrange basis polynomial L_i
        numerator   = 1.0
        denominator = 1.0
        for j in range(n):
            if j != i:
                numerator   *= (target  - x_pts[j])
                denominator *= (x_pts[i] - x_pts[j])
        L_i = numerator / denominator
        contribution = L_i * y_pts[i]
        result += contribution
        print(f"  {i+1:>2}  {L_i:>16.6f}  {y_pts[i]:>12.6f}  {contribution:>16.6f}")

    divider()
    print(f"\n  Interpolated value at x = {target} : {result:.8f}")


# ──────────────────────────────────────────────
#  CHAPTER 4 – TRAPEZOIDAL RULE
# ──────────────────────────────────────────────

def trapezoidal_rule() -> None:
    """
    Trapezoidal Rule – Approximates ∫[a,b] f(x) dx.

    Basic     : Uses a single trapezoid (two points).
    Composite : Divides [a, b] into n sub-intervals for better accuracy.
    """
    section("TRAPEZOIDAL RULE  (Numerical Integration)")

    print("  Sub-menu:")
    print("    1. Basic Trapezoidal  (1 interval)")
    print("    2. Composite Trapezoidal  (n intervals)")
    choice = input("  Select (1 or 2): ").strip()

    if choice not in ("1", "2"):
        print("  ✗ Invalid choice.")
        return

    expr = get_function("  Enter f(x)  e.g. x**2 + 1 : ")
    a    = get_float("  Lower limit a : ")
    b    = get_float("  Upper limit b : ")

    if choice == "1":
        # ── Basic ──────────────────────────────
        fa     = safe_eval(expr, a)
        fb     = safe_eval(expr, b)
        result = (b - a) / 2 * (fa + fb)
        print(f"\n  f({a}) = {fa:.6f}")
        print(f"  f({b}) = {fb:.6f}")
        divider()
        print(f"  Integral ≈ {result:.8f}  (Basic Trapezoidal, 1 interval)")

    else:
        # ── Composite ─────────────────────────
        n = get_int("  Number of intervals n : ")
        h = (b - a) / n

        print(f"\n  h = {h:.6f}")
        print(f"\n{'  i':>4}  {'x_i':>12}  {'f(x_i)':>14}")
        divider()

        total = 0.0
        for i in range(n + 1):
            xi  = a + i * h
            fxi = safe_eval(expr, xi)

            # Endpoints get weight 1; interior points get weight 2
            weight = 1 if (i == 0 or i == n) else 2
            total += weight * fxi
            print(f"  {i:>4}  {xi:>12.6f}  {fxi:>14.6f}" +
                  ("  (endpoint)" if weight == 1 else ""))

        result = (h / 2) * total
        divider()
        print(f"\n  Integral ≈ {result:.8f}  (Composite Trapezoidal, {n} intervals)")


# ──────────────────────────────────────────────
#  MAIN MENU
# ──────────────────────────────────────────────

MENU = """
╔══════════════════════════════════════════════════╗
║         NUMERICAL METHODS – MAIN MENU            ║
╠══════════════════════════════════════════════════╣
║  1. Bisection Method        (Root Finding)       ║
║  2. Jacobi Method           (Linear Systems)     ║
║  3. Gauss-Seidel Method     (Linear Systems)     ║
║  4. Lagrange Interpolation  (Interpolation)      ║
║  5. Trapezoidal Rule        (Integration)        ║
║  6. Exit                                         ║
╚══════════════════════════════════════════════════╝
"""

DISPATCH = {
    "1": bisection_method,
    "2": jacobi_method,
    "3": gauss_seidel_method,
    "4": lagrange_interpolation,
    "5": trapezoidal_rule,
}


def main() -> None:
    print("\n  Welcome to the Numerical Methods Toolkit!")

    while True:
        print(MENU)
        choice = input("  Your choice (1-6): ").strip()

        if choice == "6":
            print("\n  Goodbye! 👋\n")
            break
        elif choice in DISPATCH:
            try:
                DISPATCH[choice]()
            except KeyboardInterrupt:
                print("\n  (Interrupted – returning to main menu.)")
            except Exception as e:
                print(f"\n  ✗ Unexpected error: {e}")
        else:
            print("  ✗ Invalid choice. Please enter 1–6.")


if __name__ == "__main__":
    main()