import math
import itertools
import tkinter as tk
from tkinter import ttk, messagebox #error popup

# ── Matplotlib (embedded) ─────────────────────────────────────
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #graph in tkinter
import numpy as np


# ════════════════════════════════════════════════════════════════
#  DARK THEME CONSTANTS
# ════════════════════════════════════════════════════════════════

BG         = "#0f1117"
PANEL_BG   = "#1a1d27"
CARD_BG    = "#20243a"
ACCENT     = "#4f8ef7"
ACCENT2    = "#7c5cbf"
ACCENT_RUN = "#2db87a"
TEXT       = "#e8eaf0"
TEXT_DIM   = "#8b90a0"
BORDER     = "#2a2d3a"
SUCCESS    = "#3ecf8e"
WARN_COL   = "#f0b429"
ERROR_COL  = "#f76f72"
MONO       = ("Courier New", 10)
BODY       = ("Segoe UI", 10)
BODY_BOLD  = ("Segoe UI", 10, "bold")
HEADING    = ("Segoe UI", 13, "bold")
TITLE_FONT = ("Segoe UI", 15, "bold")
HINT_FONT  = ("Segoe UI", 8)
ENTRY_BG   = "#252837"

# Matplotlib dark palette
MPL_BG     = "#1a1d27"
MPL_FG     = "#e8eaf0"
MPL_GRID   = "#2a2d3a"
MPL_ACCENT = "#4f8ef7"
MPL_GREEN  = "#3ecf8e"
MPL_RED    = "#f76f72"
MPL_AMBER  = "#f0b429"
MPL_PURPLE = "#7c5cbf"


# ════════════════════════════════════════════════════════════════
#  CORE NUMERICAL ALGORITHMS 
# ════════════════════════════════════════════════════════════════

def safe_eval(expr: str, x: float) -> float:
    """Take equation written by user as text and convert it into real mathematical calculation."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["x"] = x
    return eval(expr, {"__builtins__": {}}, allowed)


# ── Ch.1 · Bisection ─────────────────────────────────────────
def run_bisection(expr, a, b, tol, max_iter):
    """Returns (lines, root, iterations, converged)."""
    fa = safe_eval(expr, a)
    fb = safe_eval(expr, b)
    if fa * fb >= 0:     #if signs not opp give error
        raise ValueError(
            f"f(a) and f(b) must have opposite signs.\n"
            f"  f({a}) = {fa:.6f}\n"
            f"  f({b}) = {fb:.6f}\n"
            f"Choose a and b so that f(a) × f(b) < 0."
        )
    lines = []
    lines.append(f"{'Iter':>4}  {'a':>12}  {'b':>12}  {'c (midpoint)':>14}  {'f(c)':>12}")
    lines.append("─" * 62)
    root      = None
    converged = False
    iterations = 0
    a0, b0 = a, b   # keep originals for graph

    for i in range(1, max_iter + 1): #until tol or max iter
        c  = (a + b) / 2
        fc = safe_eval(expr, c)
        lines.append(f"{i:>4}  {a:>12.6f}  {b:>12.6f}  {c:>14.6f}  {fc:>12.6f}")
        iterations = i
        if abs(fc) < tol or (b - a) / 2 < tol: #check conv
            root      = c
            converged = True
            break
        if safe_eval(expr, a) * fc < 0:
            b = c
        else:
            a = c
    else:
        root = (a + b) / 2

    lines.append("─" * 62)
    if converged:
        lines.append(f"\n  ✔  Converged!  Root ≈ {root:.10f}")
    else:
        lines.append(f"\n  ⚠  Max iterations reached without full convergence.")
        lines.append(f"     Best estimate: Root ≈ {root:.10f}")
    lines.append(f"  Iterations used : {iterations}")
    return lines, root, iterations, converged, a0, b0


# ── Ch.2 · Jacobi ────────────────────────────────────────────
def run_jacobi(A, b, x0, tol, max_iter):
    """Returns (lines, solution, iterations, converged)."""
    n = len(b)
    x = x0[:]
    lines = []
    header = f"{'Iter':>4}  " + "  ".join(f"{'x'+str(i+1):>12}" for i in range(n))
    lines.append(header)
    lines.append("─" * max(62, len(header) + 4))
    iterations = 0
    converged  = False
    for it in range(1, max_iter + 1):
        x_new = []
        for i in range(n):
            if A[i][i] == 0:
                raise ValueError(
                    f"Division by zero: diagonal element A[{i+1}][{i+1}] = 0.\n"
                    "Jacobi cannot proceed with a zero diagonal."
                )
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new.append((b[i] - sigma) / A[i][i])
        lines.append(f"{it:>4}  " + "  ".join(f"{v:>12.6f}" for v in x_new))
        iterations = it
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            x         = x_new
            converged = True
            break
        x = x_new
    lines.append("─" * max(62, len(header) + 4))
    lines.append("\n  ✔  Solution vector x:") if converged else \
    lines.append("\n  ⚠  Method did not converge within maximum iterations.\n     Best estimate:")
    for i, v in enumerate(x):
        lines.append(f"       x[{i+1}] = {v:.10f}")
    lines.append(f"  Iterations used : {iterations}")
    return lines, x, iterations, converged


# ── Ch.2 · Gauss-Seidel ──────────────────────────────────────
def run_gauss_seidel(A, b, x0, tol, max_iter):
    """Returns (lines, solution, iterations, converged)."""
    n = len(b)
    x = x0[:]
    lines = []
    header = f"{'Iter':>4}  " + "  ".join(f"{'x'+str(i+1):>12}" for i in range(n))
    lines.append(header)
    lines.append("─" * max(62, len(header) + 4))
    iterations = 0
    converged  = False
    for it in range(1, max_iter + 1):
        x_old = x[:]
        for i in range(n):
            if A[i][i] == 0: #cant divide by zero
                raise ValueError(
                    f"Division by zero: diagonal element A[{i+1}][{i+1}] = 0.\n"
                    "Gauss-Seidel cannot proceed with a zero diagonal."
                )
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]
        lines.append(f"{it:>4}  " + "  ".join(f"{v:>12.6f}" for v in x))
        iterations = it
        if max(abs(x[i] - x_old[i]) for i in range(n)) < tol: #conv true
            converged = True
            break
    lines.append("─" * max(62, len(header) + 4))
    lines.append("\n  ✔  Solution vector x:") if converged else \
    lines.append("\n  ⚠  Method did not converge within maximum iterations.\n     Best estimate:")
    for i, v in enumerate(x):
        lines.append(f"       x[{i+1}] = {v:.10f}")
    lines.append(f"  Iterations used : {iterations}")
    return lines, x, iterations, converged


# ── Ch.3 · Lagrange ──────────────────────────────────────────
def run_lagrange(x_pts, y_pts, target):
    """Returns (lines, result)."""
    n = len(x_pts)
    if len(set(x_pts)) != n:
        raise ValueError("All x values must be distinct (no duplicates allowed).")
    lines = []
    lines.append(f"{'Term':>4}  {'L_i(x)':>16}  {'y_i':>12}  {'Contribution':>16}")
    lines.append("─" * 56)
    result = 0.0
    for i in range(n):
        num = den = 1.0
        for j in range(n):
            if j != i:
                num *= (target   - x_pts[j])
                den *= (x_pts[i] - x_pts[j])
        L_i          = num / den
        contribution = L_i * y_pts[i]
        result      += contribution
        lines.append(f"  {i+1:>2}  {L_i:>16.6f}  {y_pts[i]:>12.6f}  {contribution:>16.6f}")
    lines.append("─" * 56)
    lines.append(f"\n  ✔  Interpolated value at x = {target} : {result:.10f}")
    return lines, result


# ── Ch.4 · Trapezoidal (basic) ───────────────────────────────
def run_trapezoidal_basic(expr, a, b):
    """Returns (lines, result)."""
    fa = safe_eval(expr, a)
    fb = safe_eval(expr, b)
    result = (b - a) / 2 * (fa + fb)
    lines = [
        f"  f({a}) = {fa:.8f}",
        f"  f({b}) = {fb:.8f}",
        "─" * 42,
        f"\n  ✔  Integral ≈ {result:.10f}  (Basic, 1 interval)",
    ]
    return lines, result


# ── Ch.4 · Trapezoidal (composite) ───────────────────────────
def run_trapezoidal_composite(expr, a, b, n):
    """Returns (lines, result)."""
    h = (b - a) / n
    lines = [
        f"  Step size h = {h:.8f}\n",
        f"{'  i':>4}  {'x_i':>14}  {'f(x_i)':>16}  {'weight':>8}",
        "─" * 50,
    ]
    total = 0.0
    for i in range(n + 1):
        xi     = a + i * h
        fxi    = safe_eval(expr, xi)
        weight = 1 if (i == 0 or i == n) else 2
        total += weight * fxi
        tag    = " ← endpoint" if weight == 1 else ""
        lines.append(f"  {i:>4}  {xi:>14.6f}  {fxi:>16.6f}  {weight:>8}{tag}")
    result = (h / 2) * total
    lines.append("─" * 50)
    lines.append(f"\n  ✔  Integral ≈ {result:.10f}  (Composite, {n} intervals)")
    return lines, result


# ════════════════════════════════════════════════════════════════
#  DIAGONAL DOMINANCE   (UI)
# ════════════════════════════════════════════════════════════════

def is_diagonally_dominant(A):
    """True if |a_ii| ≥ Σ|a_ij|  for every row i."""
    for i in range(len(A)):
        if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(len(A)) if j != i):
            return False
    return True


def try_make_dominant(A, b):
    """
    Try all row permutations to find a diagonally-dominant ordering.
    Returns (A_new, b_new, success, perm_used).
    """
    n = len(A)
    for perm in itertools.permutations(range(n)):
        A_new = [A[i][:] for i in perm]
        b_new = [b[i]    for i in perm]
        if is_diagonally_dominant(A_new):
            return A_new, b_new, True, list(perm)
    return A, b, False, list(range(n))


# ════════════════════════════════════════════════════════════════
#  WIDGET FACTORIES
# ════════════════════════════════════════════════════════════════

def styled_label(parent, text, font=BODY, color=TEXT_DIM, **kw):
    return tk.Label(parent, text=text, font=font, fg=color, bg=PANEL_BG, **kw)


def hint_label(parent, text, bg=PANEL_BG):
    return tk.Label(parent, text=text, font=HINT_FONT, fg="#5a6070",
                    bg=bg, anchor="w")


def make_ph_entry(parent, placeholder, width=28, fg_active=TEXT,
                  bg=ENTRY_BG, font=BODY):
    """
    Entry with placeholder behaviour.
    When showing placeholder → fg is TEXT_DIM, get_value() returns "".
    When user has typed     → fg is fg_active,  get_value() returns text.
    """
    e = tk.Entry(parent, font=font, bg=bg, fg=TEXT_DIM,
                 insertbackground=ACCENT, relief="flat",
                 highlightthickness=1, highlightcolor=ACCENT,
                 highlightbackground=BORDER, width=width)
    e._ph   = placeholder
    e._show = True
    e.insert(0, placeholder)

    def _in(ev, w=e):
        if w._show:
            w.delete(0, "end")
            w.config(fg=fg_active)
            w._show = False

    def _out(ev, w=e):
        if w.get().strip() == "":
            w.delete(0, "end")
            w.insert(0, w._ph)
            w.config(fg=TEXT_DIM)
            w._show = True

    e.bind("<FocusIn>",  _in)
    e.bind("<FocusOut>", _out)

    def get_value(w=e):
        return "" if w._show else w.get()

    e.get_value = get_value
    return e


def ph_row(parent, label_text, placeholder, hint="", row=0, col=0,
           width=28, colspan_hint=1):
    """Label + placeholder Entry in a grid. Returns the Entry."""
    styled_label(parent, label_text).grid(
        row=row, column=col, sticky="w", padx=(0, 8), pady=5)
    e = make_ph_entry(parent, placeholder, width=width)
    e.grid(row=row, column=col + 1, sticky="w", pady=5)
    if hint:
        hint_label(parent, hint).grid(
            row=row + 1, column=col + 1, sticky="w", pady=0,
            columnspan=colspan_hint)
    return e


def matrix_cell(parent, row, col, placeholder="0"):
    """Compact placeholder cell for the A matrix or b/x0 vectors."""
    e = make_ph_entry(parent, placeholder, width=7,
                      fg_active=TEXT, font=MONO)
    e.grid(row=row, column=col, padx=3, pady=3)
    return e


# ── Input readers (raise ValueError with clear messages) ─────

def gf(entry, name):
    """Read float from a placeholder entry."""
    v = (entry.get_value() if hasattr(entry, "get_value") else entry.get()).strip()
    if not v:
        raise ValueError(f'"{name}" is required.')
    try:
        return float(v)
    except ValueError:
        raise ValueError(f'"{name}" must be a number.\nYou entered: "{v}"')


def gi(entry, name, min_val=1):
    """Read int ≥ min_val from a placeholder entry."""
    v = (entry.get_value() if hasattr(entry, "get_value") else entry.get()).strip()
    if not v:
        raise ValueError(f'"{name}" is required.')
    try:
        val = int(v)
    except ValueError:
        raise ValueError(f'"{name}" must be a whole number.\nYou entered: "{v}"')
    if val < min_val:
        raise ValueError(f'"{name}" must be ≥ {min_val}.')
    return val


def ge(entry, name):
    """Read and validate a math expression from a placeholder entry."""
    v = (entry.get_value() if hasattr(entry, "get_value") else entry.get()).strip()
    if not v:
        raise ValueError(f'"{name}" cannot be empty.\nExample: x**2 + 3*x - 4')
    try:
        safe_eval(v, 1.0)
    except ZeroDivisionError:
        raise ValueError(
            f'Division by zero evaluating "{name}" at x = 1.\n'
            f'Expression: {v}')
    except Exception as exc:
        raise ValueError(
            f'Invalid expression in "{name}".\n'
            f'Expression : {v}\n'
            f'Detail     : {exc}\n'
            f'Hint       : use Python syntax, e.g.  x**2 + 3*x  or  sin(x)')
    return v


def validate_tol(entry, name="Tolerance"):
    tol = gf(entry, name)
    if tol <= 0:
        raise ValueError(
            f'"{name}" must be strictly positive (> 0).\n'
            f'You entered: {tol}\n'
            f'Example: 0.0001  or  1e-6')
    return tol


# ════════════════════════════════════════════════════════════════
#  MATPLOTLIB GRAPH HELPERS  
# ════════════════════════════════════════════════════════════════

def dark_fig(ax):
    """Apply dark theme styling to a matplotlib Figure / Axes."""
    ax.figure.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_BG)
    ax.tick_params(colors=MPL_FG, which="both")
    ax.xaxis.label.set_color(MPL_FG)
    ax.yaxis.label.set_color(MPL_FG)
    ax.title.set_color(MPL_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(MPL_GRID)
    ax.grid(True, color=MPL_GRID, linewidth=0.6, linestyle="--")


def plot_bisection(fig, ax, expr, a, b, root):
    """Plot f(x) over [a,b], highlight interval and root."""
    ax.clear()
    margin  = (b - a) * 0.4
    xs      = np.linspace(a - margin, b + margin, 500)
    ys      = [safe_eval(expr, x) for x in xs]
    ax.plot(xs, ys, color=MPL_ACCENT, linewidth=2, label=f"f(x) = {expr}")
    ax.axhline(0, color=MPL_FG, linewidth=0.8, linestyle="-")
    ax.axvspan(a, b, alpha=0.12, color=MPL_AMBER, label=f"Interval [{a}, {b}]")
    ax.axvline(a, color=MPL_AMBER, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvline(b, color=MPL_AMBER, linewidth=1.2, linestyle="--", alpha=0.7)
    if root is not None:
        ax.scatter([root], [0], color=MPL_RED, zorder=5, s=80,
                   label=f"Root ≈ {root:.6f}")
        ax.axvline(root, color=MPL_RED, linewidth=1, linestyle=":", alpha=0.8)
    dark_fig(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Bisection Method  —  Root Finding", pad=10)
    ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER,
              labelcolor=MPL_FG, loc="best")
    fig.tight_layout()


def plot_lagrange(fig, ax, x_pts, y_pts, target, result):
    """Plot Lagrange interpolating polynomial with data points."""
    ax.clear()
    xmin, xmax = min(x_pts), max(x_pts)
    margin = (xmax - xmin) * 0.15 or 0.5
    xs     = np.linspace(xmin - margin, xmax + margin, 600)

    def lagrange_poly(t):
        n  = len(x_pts)
        s  = 0.0
        for i in range(n):
            L = 1.0
            for j in range(n):
                if j != i:
                    L *= (t - x_pts[j]) / (x_pts[i] - x_pts[j])
            s += y_pts[i] * L
        return s

    ys = [lagrange_poly(t) for t in xs]
    ax.plot(xs, ys, color=MPL_ACCENT, linewidth=2,
            label="Lagrange Polynomial")
    ax.scatter(x_pts, y_pts, color=MPL_GREEN, zorder=5, s=70,
               label="Data Points", edgecolors=MPL_BG, linewidths=0.8)
    ax.scatter([target], [result], color=MPL_RED, zorder=6, s=90,
               marker="*", label=f"P({target}) ≈ {result:.4f}")
    ax.axvline(target, color=MPL_RED, linewidth=1, linestyle=":", alpha=0.7)
    ax.axhline(result, color=MPL_RED, linewidth=1, linestyle=":", alpha=0.7)
    dark_fig(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
    ax.set_title("Lagrange Interpolation", pad=10)
    ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER,
              labelcolor=MPL_FG, loc="best")
    fig.tight_layout()


def plot_trapezoidal(fig, ax, expr, a, b, mode, n_intervals=1):
    """Plot f(x) with shaded trapezoid regions."""
    ax.clear()
    margin = (b - a) * 0.15
    xs     = np.linspace(a - margin, b + margin, 600)
    ys     = [safe_eval(expr, t) for t in xs]
    ax.plot(xs, ys, color=MPL_ACCENT, linewidth=2,
            label=f"f(x) = {expr}", zorder=3)
    ax.axhline(0, color=MPL_FG, linewidth=0.7)

    n = n_intervals if mode == "composite" else 1
    h = (b - a) / n
    for k in range(n):
        x0 = a + k * h
        x1 = a + (k + 1) * h
        fx0 = safe_eval(expr, x0)
        fx1 = safe_eval(expr, x1)
        px  = [x0, x1, x1, x0, x0]
        py  = [0,  0,  fx1, fx0, 0]
        ax.fill(px, py, alpha=0.25, color=MPL_PURPLE)
        ax.plot([x0, x0], [0, fx0], color=MPL_AMBER,
                linewidth=0.9, linestyle="--")
        ax.plot([x1, x1], [0, fx1], color=MPL_AMBER,
                linewidth=0.9, linestyle="--")
        ax.plot([x0, x1], [fx0, fx1], color=MPL_AMBER,
                linewidth=1.2, linestyle="-",
                label="Trapezoid tops" if k == 0 else "")

    dark_fig(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    title = "Trapezoidal Rule — " + (
        "Basic (1 interval)" if mode == "basic"
        else f"Composite ({n} intervals)")
    ax.set_title(title, pad=10)
    ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER,
              labelcolor=MPL_FG, loc="best")
    fig.tight_layout()


# ════════════════════════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ════════════════════════════════════════════════════════════════

class NumericalMethodsApp(tk.Tk):

    METHODS = [
        ("Ch.1  Bisection",    "bisection"),
        ("Ch.2  Jacobi",       "jacobi"),
        ("Ch.2  Gauss-Seidel", "gauss_seidel"),
        ("Ch.3  Lagrange",     "lagrange"),
        ("Ch.4  Trapezoidal",  "trapezoidal"),
    ]

    def __init__(self):
        super().__init__()
        self.title("Numerical Methods Project")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(1000, 660)
        self.geometry("1200x780")

        self._sidebar_btns = {}
        self._fig          = None
        self._ax           = None
        self._mpl_canvas   = None

        self._build_skeleton()
        self._show_panel("bisection")

    # ── skeleton ────────────────────────────────────────────────
    def _build_skeleton(self):
        # title bar
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x")
        tk.Label(bar, text="⚙  Numerical Methods Project",
                 font=TITLE_FONT, fg=ACCENT, bg=BG,
                 pady=12, padx=20).pack(side="left")
       # tk.Label(bar,
        #         text="Root Finding · Linear Systems · Interpolation · Integration",
         #        font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG).pack(side="left")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # sidebar
        self._sidebar = tk.Frame(body, bg=PANEL_BG, width=195)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)
        self._build_sidebar()
        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        # ── right column  (three panes stacked vertically) ──
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # configure 3 rows: input fixed, graph flexible, output flexible
        right.rowconfigure(0, weight=0)   # inputs  – shrink-wrapped
        right.rowconfigure(1, weight=2)   # graph   – grows
        right.rowconfigure(2, weight=3)   # output  – grows more
        right.columnconfigure(0, weight=1)

        # ── Zone 0: Input (scrollable inner frame) ──────────────
        input_outer = tk.Frame(right, bg=PANEL_BG,
                               highlightthickness=1, highlightbackground=BORDER)
        input_outer.grid(row=0, column=0, sticky="ew",
                         padx=16, pady=14)

        self._input_canvas = tk.Canvas(input_outer, bg=PANEL_BG,
                                       highlightthickness=0)
        input_vsb = ttk.Scrollbar(input_outer, orient="vertical",
                                  command=self._input_canvas.yview)
        self._input_canvas.configure(yscrollcommand=input_vsb.set)
        input_vsb.pack(side="right", fill="y")
        self._input_canvas.pack(side="left", fill="both", expand=True)

        self._input_frame = tk.Frame(self._input_canvas, bg=PANEL_BG)
        self._icf_id = self._input_canvas.create_window(
            (0, 0), window=self._input_frame, anchor="nw")

        def _icf_resize(e):
            self._input_canvas.configure(scrollregion=self._input_canvas.bbox("all"))
            h = min(self._input_frame.winfo_reqheight(), 300)
            self._input_canvas.configure(height=h)

        def _ic_resize(e):
            self._input_canvas.itemconfig(self._icf_id, width=e.width)

        self._input_frame.bind("<Configure>", _icf_resize)
        self._input_canvas.bind("<Configure>", _ic_resize)

        # mouse-wheel scroll on the input zone
        def _wheel(e, c=self._input_canvas):
            c.yview_scroll(int(-1 * (e.delta / 120)), "units")

        self._input_canvas.bind("<MouseWheel>", _wheel)
        self._input_frame.bind("<MouseWheel>",  _wheel)

        # ── Zone 1: Graph ────────────────────────────────────────
        graph_outer = tk.Frame(right, bg=PANEL_BG,
                               highlightthickness=1, highlightbackground=BORDER)
        graph_outer.grid(row=1, column=0, sticky="nsew",
                         padx=16, pady=8)

        graph_header = tk.Frame(graph_outer, bg=PANEL_BG)
        graph_header.pack(fill="x")
        tk.Label(graph_header, text="GRAPH VISUALIZATION",
                 font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=PANEL_BG, padx=10, pady=4).pack(side="left")
        self._graph_note = tk.Label(graph_header, text="",
                                    font=HINT_FONT, fg=TEXT_DIM, bg=PANEL_BG)
        self._graph_note.pack(side="left")

        self._graph_frame = tk.Frame(graph_outer, bg=PANEL_BG)
        self._graph_frame.pack(fill="both", expand=True)

        # placeholder when no graph exists yet
        self._graph_placeholder = tk.Label(
            self._graph_frame,
            text="Graph appears here after pressing Run",
            font=BODY, fg=TEXT_DIM, bg=PANEL_BG)
        self._graph_placeholder.pack(expand=True)

        # ── Zone 2: Output console ───────────────────────────────
        out_outer = tk.Frame(right, bg=PANEL_BG,
                             highlightthickness=1, highlightbackground=BORDER)
        out_outer.grid(row=2, column=0, sticky="nsew",
                       padx=16, pady=8)

        tk.Label(out_outer, text="OUTPUT CONSOLE",
                 font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=PANEL_BG,
                 padx=10, pady=4, anchor="w").pack(fill="x")

        self._output = tk.Text(out_outer, font=MONO, bg="#11141f", fg=TEXT,
                               relief="flat", padx=14, pady=10,
                               state="disabled", wrap="none",
                               selectbackground=ACCENT2)
        vsb = ttk.Scrollbar(out_outer, orient="vertical",   command=self._output.yview)
        hsb = ttk.Scrollbar(out_outer, orient="horizontal", command=self._output.xview)
        self._output.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._output.pack(fill="both", expand=True)

        # output text tags
        self._output.tag_configure("title",  foreground=ACCENT,
                                   font=(*MONO[:1], 11, "bold"))
        self._output.tag_configure("result", foreground=SUCCESS,
                                   font=(*MONO[:1], 10, "bold"))
        self._output.tag_configure("warn",   foreground=WARN_COL,
                                   font=(*MONO[:1], 10, "italic"))
        self._output.tag_configure("info",   foreground=WARN_COL)
        self._output.tag_configure("dim",    foreground=TEXT_DIM)
        self._output.tag_configure("normal", foreground=TEXT)

    # ── sidebar ─────────────────────────────────────────────────
    def _build_sidebar(self):
        tk.Label(self._sidebar, text="METHODS",
                 font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg=PANEL_BG, anchor="w",
                 padx=16, pady=12).pack(fill="x")
        for label, key in self.METHODS:
            btn = tk.Button(
                self._sidebar, text=label, font=("Segoe UI", 10),
                fg=TEXT, bg=PANEL_BG, activebackground=ACCENT2,
                activeforeground="white", relief="flat",
                anchor="w", padx=16, pady=10, cursor="hand2",
                command=lambda k=key: self._show_panel(k))
            btn.pack(fill="x")
            self._sidebar_btns[key] = btn

    def _hl_sidebar(self, key):
        for k, btn in self._sidebar_btns.items():
            btn.config(bg=ACCENT2 if k == key else PANEL_BG,
                       fg="white" if k == key else TEXT)

    # ── panel switching ──────────────────────────────────────────
    def _show_panel(self, key):
        self._hl_sidebar(key)
        self._clear_inputs()
        self._clear_output()
        self._clear_graph()
        {
            "bisection":    self._panel_bisection,
            "jacobi":       self._panel_jacobi,
            "gauss_seidel": self._panel_gauss_seidel,
            "lagrange":     self._panel_lagrange,
            "trapezoidal":  self._panel_trapezoidal,
        }[key]()

    # ── zone helpers ─────────────────────────────────────────────
    def _clear_inputs(self):
        for w in self._input_frame.winfo_children():
            w.destroy()

    def _clear_output(self):
        self._output.config(state="normal")
        self._output.delete("1.0", "end")
        self._output.config(state="disabled")

    def _clear_graph(self):
        for w in self._graph_frame.winfo_children():
            w.destroy()
        self._graph_placeholder = tk.Label(
            self._graph_frame,
            text="Graph appears here after pressing Run",
            font=BODY, fg=TEXT_DIM, bg=PANEL_BG)
        self._graph_placeholder.pack(expand=True)
        self._graph_note.config(text="")
        self._fig = self._ax = self._mpl_canvas = None

    def _ensure_graph(self):
        """Create or reuse the matplotlib figure inside the graph zone."""
        if self._mpl_canvas is None:
            for w in self._graph_frame.winfo_children():
                w.destroy()
            self._fig, self._ax = plt.subplots(figsize=(5, 2.6),
                                               facecolor=MPL_BG)
            self._mpl_canvas = FigureCanvasTkAgg(
                self._fig, master=self._graph_frame)
            self._mpl_canvas.get_tk_widget().pack(
                fill="both", expand=True, padx=4, pady=4)
        return self._fig, self._ax

    def _refresh_graph(self):
        if self._mpl_canvas:
            self._mpl_canvas.draw()

    def _write_output(self, lines):
        """Write formatted lines to the output console."""
        self._output.config(state="normal")
        self._output.delete("1.0", "end")
        for raw in lines:
            s = raw.strip()
            if s.startswith("✔"):
                tag = "result"
            elif s.startswith(("⚠", "▶", "→")):
                tag = "warn" if "⚠" in s else "info"
            elif s.startswith(("─", "═", "━")):
                tag = "dim"
            elif (s == s.upper() and len(s) >= 4
                  and any(c.isalpha() for c in s)):
                tag = "title"
            else:
                tag = "normal"
            self._output.insert("end", raw + "\n", tag)
        self._output.config(state="disabled")
        self._output.see("1.0")

    # ── shared input-frame helpers ───────────────────────────────
    def _heading(self, title, subtitle=""):
        tk.Label(self._input_frame, text=title, font=HEADING,
                 fg=ACCENT, bg=PANEL_BG, anchor="w",
                 padx=14, pady=12).pack(fill="x")
        if subtitle:
            tk.Label(self._input_frame, text=subtitle,
                     font=("Segoe UI", 9), fg=TEXT_DIM,
                     bg=PANEL_BG, anchor="w", padx=14,
                     pady=0).pack(fill="x")
        tk.Frame(self._input_frame, bg=BORDER, height=1).pack(
            fill="x", padx=14, pady=0)

    def _grid_f(self, padx=14, pady=0):
        f = tk.Frame(self._input_frame, bg=PANEL_BG)
        f.pack(fill="x", padx=padx, pady=pady)
        return f

    def _run_btn(self, parent, text, cmd, color=ACCENT_RUN):
        return tk.Button(parent, text=text, font=BODY_BOLD,
                         fg="white", bg=color,
                         activebackground="#259960",
                         activeforeground="white",
                         relief="flat", padx=18, pady=7,
                         cursor="hand2", command=cmd)

    # ════════════════════════════════════════════════════════════
    #  PANEL 1 – BISECTION
    # ════════════════════════════════════════════════════════════
    def _panel_bisection(self):
        self._heading(
            "Bisection Method",
            "Ch.1 · Root Finding — narrows [a, b] until |f(c)| < tolerance"
        )
        f = self._grid_f()

        e_fx  = ph_row(f, "f(x)          :", "e.g.  x**3 - x - 2",
                       hint="Python syntax: x**2, sin(x), exp(x), log(x)",
                       row=0, col=0)
        e_a   = ph_row(f, "Lower bound  a:", "e.g.  1",
                       hint="f(a) × f(b) must be < 0",
                       row=2, col=0)
        e_b   = ph_row(f, "Upper bound  b:", "e.g.  2",
                       hint="Opposite sign to f(a)",
                       row=4, col=0)
        e_tol = ph_row(f, "Tolerance     :", "e.g.  0.0001",
                       hint="Must be > 0  (e.g. 0.001 or 1e-6)",
                       row=6, col=0)
        e_it  = ph_row(f, "Max iterations:", "e.g.  50",
                       hint="Maximum halving steps",
                       row=8, col=0)

        def run():
            try:
                expr = ge(e_fx,  "f(x)")
                a    = gf(e_a,   "Lower bound a")
                b    = gf(e_b,   "Upper bound b")
                tol  = validate_tol(e_tol)
                mi   = gi(e_it,  "Max iterations")
                if a >= b:
                    raise ValueError(
                        "Lower bound a must be < upper bound b.\n"
                        f"Got a = {a},  b = {b}")
                lines, root, iters, conv, a0, b0 = \
                    run_bisection(expr, a, b, tol, mi)

                self._write_output([
                    "━" * 64,
                    "  BISECTION METHOD  —  Root Finding",
                    "━" * 64,
                    f"  f(x)           = {expr}",
                    f"  Interval       = [{a0},  {b0}]",
                    f"  Tolerance      = {tol}",
                    f"  Max iterations = {mi}",
                    "─" * 64, "",
                ] + lines)

                # graph
                if not conv:
                    messagebox.showwarning(
                        "Bisection — No Full Convergence",
                        f"Max iterations ({mi}) reached before tolerance was met.\n"
                        f"Best root estimate: {root:.8f}")
                fig, ax = self._ensure_graph()
                plot_bisection(fig, ax, expr, a0, b0, root)
                self._refresh_graph()
                self._graph_note.config(
                    text=f"  f(x) = {expr}   root ≈ {root:.6f}")

            except Exception as ex:
                messagebox.showerror("Bisection — Error", str(ex))

        btn_f = tk.Frame(f, bg=PANEL_BG)
        btn_f.grid(row=10, column=0, columnspan=3, sticky="w", pady=12)
        self._run_btn(btn_f, "▶  Run Bisection", run).pack(side="left")

    # ════════════════════════════════════════════════════════════
    #  SHARED MATRIX BUILDER  (Jacobi + Gauss-Seidel)
    # ════════════════════════════════════════════════════════════
    def _matrix_panel(self, title, subtitle, algo_fn, method_label):
        self._heading(title, subtitle)
        outer = self._grid_f(pady=4)

        # n row
        top = tk.Frame(outer, bg=PANEL_BG)
        top.pack(fill="x", pady=0)
        styled_label(top, "Equations  n :").pack(side="left", padx=(0, 6))
        e_n = make_ph_entry(top, "e.g.  3", width=10)
        e_n.pack(side="left", padx=(0, 10))
        hint_label(top, "Then click  Build →").pack(side="left", padx=(0, 14))

        build_btn = tk.Button(
            top, text="Build Matrix Fields →",
            font=BODY, fg=TEXT, bg="#2e3145",
            activebackground=ACCENT2, activeforeground="white",
            relief="flat", padx=10, pady=4, cursor="hand2")
        build_btn.pack(side="left")

        hint_label(outer,
                   "ℹ  Enter coefficients row by row.  b column = right-hand constants."
                   ).pack(anchor="w", pady=0)
        hint_label(outer,
                   "ℹ  Click a cell to clear its placeholder and type your value."
                   ).pack(anchor="w", pady=0)

        matrix_area = tk.Frame(outer, bg=PANEL_BG)
        matrix_area.pack(fill="x", pady=2)

        state = {"A": [], "b_e": [], "x0": [], "tol_e": None, "it_e": None}

        def build():
            for w in matrix_area.winfo_children():
                w.destroy()
            state["A"] = []; state["b_e"] = []; state["x0"] = []
            try:
                n = gi(e_n, "n")
            except Exception as ex:
                messagebox.showerror("Input Error", str(ex)); return

            # column headers
            for j in range(n):
                tk.Label(matrix_area, text=f" x{j+1} ",
                         font=HINT_FONT, fg=ACCENT, bg=PANEL_BG
                         ).grid(row=0, column=j, padx=3)
            tk.Label(matrix_area, text="  |  b",
                     font=HINT_FONT, fg=SUCCESS, bg=PANEL_BG
                     ).grid(row=0, column=n, columnspan=2, padx=3)

            # A rows + b column
            for i in range(n):
                row_e = [matrix_cell(matrix_area, i + 1, j, f"a{i+1}{j+1}")
                         for j in range(n)]
                state["A"].append(row_e)
                tk.Label(matrix_area, text=" | ", font=MONO,
                         fg=TEXT_DIM, bg=PANEL_BG
                         ).grid(row=i + 1, column=n)
                be = matrix_cell(matrix_area, i + 1, n + 1, f"b{i+1}")
                state["b_e"].append(be)

            # initial guesses
            tk.Label(matrix_area, text="\nInitial Guesses  x₀",
                     font=BODY_BOLD, fg=TEXT, bg=PANEL_BG
                     ).grid(row=n + 2, column=0, columnspan=n + 2,
                            sticky="w", pady=8)
            hint_label(matrix_area,
                       "Starting approximation for each variable"
                       ).grid(row=n + 3, column=0, columnspan=n + 2,
                              sticky="w", pady=0)
            for i in range(n):
                xe = matrix_cell(matrix_area, n + 4, i, f"x{i+1}₀")
                state["x0"].append(xe)

            # tol + iterations
            r = n + 6
            styled_label(matrix_area, "Tolerance :").grid(
                row=r, column=0, sticky="w", pady=4, columnspan=2)
            te = make_ph_entry(matrix_area, "e.g.  0.0001", width=16)
            te.grid(row=r, column=2, sticky="w", pady=4, columnspan=max(1, n - 1))
            hint_label(matrix_area, "Convergence criterion  (must be > 0)").grid(
                row=r + 1, column=2, sticky="w",
                columnspan=max(1, n - 1))
            state["tol_e"] = te

            styled_label(matrix_area, "Max iterations :").grid(
                row=r + 2, column=0, sticky="w", pady=4, columnspan=2)
            ie = make_ph_entry(matrix_area, "e.g.  100", width=16)
            ie.grid(row=r + 2, column=2, sticky="w", pady=4,
                    columnspan=max(1, n - 1))
            hint_label(matrix_area, "Stop after this many iterations").grid(
                row=r + 3, column=2, sticky="w",
                columnspan=max(1, n - 1))
            state["it_e"] = ie

            self._run_btn(matrix_area, f"▶  Run {method_label}", run,
                          color=ACCENT_RUN
                          ).grid(row=r + 4, column=0,
                                 columnspan=max(4, n + 2),
                                 pady=14, sticky="w")

        build_btn.config(command=build)

        def run():
            try:
                n = len(state["A"])
                if n == 0:
                    raise ValueError(
                        "Please click 'Build Matrix Fields' first.")

                # read A
                A  = []
                for i in range(n):
                    row_vals = []
                    for j in range(n):
                        cell = state["A"][i][j]
                        raw  = (cell.get_value()
                                if hasattr(cell, "get_value")
                                else cell.get()).strip()
                        if not raw:
                            raise ValueError(
                                f"A[{i+1}][{j+1}] is empty.")
                        row_vals.append(float(raw))
                    A.append(row_vals)

                # read b
                bv = []
                for i, cell in enumerate(state["b_e"]):
                    raw = (cell.get_value()
                           if hasattr(cell, "get_value")
                           else cell.get()).strip()
                    if not raw:
                        raise ValueError(f"b[{i+1}] is empty.")
                    bv.append(float(raw))

                # read x0
                x0 = []
                for i, cell in enumerate(state["x0"]):
                    raw = (cell.get_value()
                           if hasattr(cell, "get_value")
                           else cell.get()).strip()
                    x0.append(float(raw) if raw else 0.0)

                tol = validate_tol(state["tol_e"])
                mi  = gi(state["it_e"], "Max iterations")

                # ── diagonal dominance check (v3) ──────────────
                dom_log = [
                    "━" * 64,
                    f"  {title.upper()}",
                    "━" * 64, "",
                    "  → Checking diagonal dominance...",
                ]

                if is_diagonally_dominant(A):
                    dom_log.append(
                        "  ✔  Matrix is diagonally dominant.  Proceeding normally.")
                    A_use, b_use = A, bv
                    perm_used    = list(range(n))
                else:
                    dom_log.append(
                        "  ⚠  Matrix is NOT diagonally dominant.")
                    dom_log.append(
                        "  → Attempting row rearrangement...")
                    A_use, b_use, ok, perm_used = try_make_dominant(A, bv)

                    if ok:
                        dom_log.append(
                            "  ✔  Matrix rearranged into diagonally dominant form.")
                        dom_log.append(
                            f"     New row order: {[p+1 for p in perm_used]}")
                        dom_log.append("")
                        dom_log.append(
                            "  Rearranged system  [A | b]:")
                        for i in range(n):
                            row_s = ("    [ "
                                     + "  ".join(f"{A_use[i][j]:>8.4f}"
                                                 for j in range(n))
                                     + f"  |  {b_use[i]:>8.4f} ]")
                            dom_log.append(row_s)
                    else:
                        dom_log += [
                            "  ✘  IMPOSSIBLE to achieve diagonal dominance.",
                            "",
                            "  Error: Matrix cannot be rearranged into diagonally",
                            "  dominant form.  Method may diverge.",
                            "  Please revise the coefficient matrix.",
                        ]
                        self._write_output(dom_log)
                        messagebox.showerror(
                            f"{method_label} — Diagonal Dominance Error",
                            "Matrix is not diagonally dominant and\n"
                            "cannot be rearranged into that form.\n\n"
                            "The method cannot guarantee convergence.\n"
                            "Please revise your coefficient matrix.")
                        return

                dom_log += [
                    "", f"  Tolerance      = {tol}",
                    f"  Max iterations = {mi}",
                    f"  Initial guess  = {x0}",
                    "─" * 64, "",
                ]

                lines, sol, iters, conv = algo_fn(A_use, b_use, x0, tol, mi)
                self._write_output(dom_log + lines)

                if not conv:
                    messagebox.showwarning(
                        f"{method_label} — No Convergence",
                        f"Method did not converge within {mi} iterations.\n"
                        "The result shown is the best estimate.\n\n"
                        "Try increasing max iterations or check\n"
                        "if the matrix is strongly diagonal dominant.")

            except ValueError as ex:
                messagebox.showerror(f"{method_label} — Input Error", str(ex))
            except Exception as ex:
                messagebox.showerror(f"{method_label} — Error", str(ex))

       # build()   # auto-build default 3×3 grid on panel open

    # ════════════════════════════════════════════════════════════
    #  PANEL 2 – JACOBI
    # ════════════════════════════════════════════════════════════
    def _panel_jacobi(self):
        self._matrix_panel(
            "Jacobi Method",
            "Ch.2 · Linear Systems — all unknowns updated simultaneously each iteration",
            run_jacobi,
            "Jacobi")

    # ════════════════════════════════════════════════════════════
    #  PANEL 3 – GAUSS-SEIDEL
    # ════════════════════════════════════════════════════════════
    def _panel_gauss_seidel(self):
        self._matrix_panel(
            "Gauss-Seidel Method",
            "Ch.2 · Linear Systems — uses the latest values immediately each step",
            run_gauss_seidel,
            "Gauss-Seidel")

    # ════════════════════════════════════════════════════════════
    #  PANEL 4 – LAGRANGE
    # ════════════════════════════════════════════════════════════
    def _panel_lagrange(self):
        self._heading(
            "Lagrange Interpolation",
            "Ch.3 · Interpolation — polynomial through n data points"
        )
        outer = self._grid_f(pady=4)

        top = tk.Frame(outer, bg=PANEL_BG)
        top.pack(fill="x", pady=0)
        styled_label(top, "Number of points  n :").pack(side="left", padx=(0, 6))
        e_n = make_ph_entry(top, "e.g.  3", width=10)
        e_n.pack(side="left", padx=(0, 10))
        hint_label(top, "Minimum 2 points").pack(side="left", padx=(0, 14))
        build_btn = tk.Button(
            top, text="Build Point Fields →",
            font=BODY, fg=TEXT, bg="#2e3145",
            activebackground=ACCENT2, activeforeground="white",
            relief="flat", padx=10, pady=4, cursor="hand2")
        build_btn.pack(side="left")

        hint_label(outer, "ℹ  All x values must be distinct.").pack(
            anchor="w", pady=0)

        pts_area = tk.Frame(outer, bg=PANEL_BG)
        pts_area.pack(fill="x", pady=2)

        state = {"x_e": [], "y_e": [], "tgt": None}

        def build():
            for w in pts_area.winfo_children():
                w.destroy()
            state["x_e"] = []; state["y_e"] = []
            try:
                n = gi(e_n, "n", min_val=2)
            except Exception as ex:
                messagebox.showerror("Input Error", str(ex)); return

            # column headers
            tk.Label(pts_area, text="Pt",
                     font=HINT_FONT, fg=TEXT_DIM, bg=PANEL_BG
                     ).grid(row=0, column=0, padx=6)
            tk.Label(pts_area, text="x value",
                     font=HINT_FONT, fg=ACCENT, bg=PANEL_BG
                     ).grid(row=0, column=1, padx=6)
            tk.Label(pts_area, text="y = f(x)",
                     font=HINT_FONT, fg=SUCCESS, bg=PANEL_BG
                     ).grid(row=0, column=2, padx=6)

            for i in range(n):
                styled_label(pts_area, f"  {i+1}").grid(
                    row=i + 1, column=0, padx=6)
                xe = make_ph_entry(pts_area, f"x{i+1}", width=12)
                xe.grid(row=i + 1, column=1, padx=4, pady=3)
                ye = make_ph_entry(pts_area, f"y{i+1}", width=12,
                                   fg_active=SUCCESS)
                ye.grid(row=i + 1, column=2, padx=4, pady=3)
                state["x_e"].append(xe)
                state["y_e"].append(ye)

            r = n + 2
            tk.Label(pts_area, text="  Interpolate at  x =",
                     font=BODY, fg=TEXT_DIM, bg=PANEL_BG
                     ).grid(row=r, column=0, columnspan=2,
                            sticky="w", pady=12, padx=4)
            te = make_ph_entry(pts_area, "e.g.  2.5", width=14,
                               fg_active=SUCCESS)
            te.grid(row=r, column=2, padx=4, pady=12)
            hint_label(pts_area, "Value to estimate f(x) at").grid(
                row=r + 1, column=2, sticky="w")
            state["tgt"] = te

            self._run_btn(pts_area, "▶  Run Lagrange", run
                          ).grid(row=r + 2, column=0, columnspan=3,
                                 pady=14, sticky="w")

        build_btn.config(command=build)

        def run():
            try:
                x_pts  = [gf(e, f"x[{i+1}]")
                           for i, e in enumerate(state["x_e"])]
                y_pts  = [gf(e, f"y[{i+1}]")
                           for i, e in enumerate(state["y_e"])]
                target = gf(state["tgt"], "target x")

                lines, val = run_lagrange(x_pts, y_pts, target)
                self._write_output([
                    "━" * 58,
                    "  LAGRANGE INTERPOLATION",
                    "━" * 58,
                    f"  Data points   : {list(zip(x_pts, y_pts))}",
                    f"  Target x      : {target}",
                    "─" * 58, "",
                ] + lines)

                # graph
                fig, ax = self._ensure_graph()
                plot_lagrange(fig, ax, x_pts, y_pts, target, val)
                self._refresh_graph()
                self._graph_note.config(
                    text=f"  P({target}) ≈ {val:.6f}")

            except Exception as ex:
                messagebox.showerror("Lagrange — Error", str(ex))

        #build()

    # ════════════════════════════════════════════════════════════
    #  PANEL 5 – TRAPEZOIDAL
    # ════════════════════════════════════════════════════════════
    def _panel_trapezoidal(self):
        self._heading(
            "Trapezoidal Rule",
            "Ch.4 · Numerical Integration — approximate  ∫ f(x) dx  from a to b"
        )
        f = self._grid_f(pady=4)

        # shared inputs  (compact 2-row layout)
        e_fx = ph_row(f, "f(x)         :", "e.g.  x**2 + 1",
                      hint="Python syntax: x**2, sin(x), exp(x)",
                      row=0, col=0)
        e_a  = ph_row(f, "Lower limit a:", "e.g.  0",
                      hint="Left endpoint of integration",
                      row=2, col=0)
        e_b  = ph_row(f, "Upper limit b:", "e.g.  1",
                      hint="Right endpoint  (must be > a)",
                      row=4, col=0)

        # n (composite only)
        styled_label(f, "Intervals  n :").grid(
            row=6, column=0, sticky="w", padx=(0, 8), pady=5)
        e_n = make_ph_entry(f, "e.g.  10  (composite only)", width=30)
        e_n.grid(row=6, column=1, sticky="w", pady=5)
        hint_label(f, "Number of sub-intervals — only used for Composite mode").grid(
            row=7, column=1, sticky="w", pady=0)

        # separator
        tk.Frame(f, bg=BORDER, height=1).grid(
            row=8, column=0, columnspan=3, sticky="ew", pady=10)

        # ── common reader ──
        def _read():
            expr = ge(e_fx, "f(x)")
            a    = gf(e_a,  "Lower limit a")
            b    = gf(e_b,  "Upper limit b")
            if a >= b:
                raise ValueError(
                    f"Lower limit a must be < upper limit b.\n"
                    f"Got a = {a},  b = {b}")
            return expr, a, b

        # ── Run Basic ──
        def run_basic():
            try:
                expr, a, b = _read()
                lines, result = run_trapezoidal_basic(expr, a, b)
                self._write_output([
                    "━" * 52,
                    "  TRAPEZOIDAL RULE  —  Basic (1 interval)",
                    "━" * 52,
                    f"  f(x)     = {expr}",
                    f"  Interval = [{a},  {b}]",
                    "─" * 52, "",
                ] + lines)
                fig, ax = self._ensure_graph()
                plot_trapezoidal(fig, ax, expr, a, b, "basic")
                self._refresh_graph()
                self._graph_note.config(
                    text=f"  ∫f(x)dx ≈ {result:.6f}  (Basic)")
            except Exception as ex:
                messagebox.showerror("Trapezoidal Basic — Error", str(ex))

        # ── Run Composite ──
        def run_composite():
            try:
                expr, a, b = _read()
                n = gi(e_n, "Intervals n")
                lines, result = run_trapezoidal_composite(expr, a, b, n)
                self._write_output([
                    "━" * 52,
                    f"  TRAPEZOIDAL RULE  —  Composite ({n} intervals)",
                    "━" * 52,
                    f"  f(x)      = {expr}",
                    f"  Interval  = [{a},  {b}]",
                    f"  Intervals = {n}",
                    "─" * 52, "",
                ] + lines)
                fig, ax = self._ensure_graph()
                plot_trapezoidal(fig, ax, expr, a, b, "composite", n)
                self._refresh_graph()
                self._graph_note.config(
                    text=f"  ∫f(x)dx ≈ {result:.6f}  ({n} intervals)")
            except Exception as ex:
                messagebox.showerror("Trapezoidal Composite — Error", str(ex))

        # ── TWO clearly-visible run buttons ──
        btn_row = tk.Frame(f, bg=PANEL_BG)
        btn_row.grid(row=9, column=0, columnspan=3, sticky="w", pady=4)

        self._run_btn(btn_row, "▶  Run Basic",      run_basic,
                      color=ACCENT_RUN).pack(side="left", padx=(0, 10))
        self._run_btn(btn_row, "▶  Run Composite",  run_composite,
                      color=ACCENT).pack(side="left", padx=(0, 14))
        hint_label(btn_row,
                   "Basic = 1 trapezoid   |   "
                   "Composite = n sub-intervals (fill 'Intervals n' first)",
                   bg=PANEL_BG).pack(side="left")


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = NumericalMethodsApp()
    app.mainloop()