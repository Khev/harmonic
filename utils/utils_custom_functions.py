from sympy import simplify, factor, sqrt, Pow, Basic, Integer, Mul, Add, collect, together
from sympy import Eq, ratsimp, expand, Abs, Rational, Number, sympify, symbols, integrate
from sympy import sin, cos, tan, asin, acos, atan, Symbol
from operator import add, sub, mul, truediv
from sympy import sin, cos, exp, log  # add exp, log; light types
# utils/custom_functions.py
from sympy import Pow, sqrt as sym_sqrt, simplify, count_ops
from sympy.simplify.powsimp import powdenest, powsimp
OPS_SIMPLIFY_LIMIT = 15  # try 15–30; lower = safer/faster
from sympy import S  # Add this import if not already present

# utils/safe_eval.py
import signal
from contextlib import contextmanager

class SympyTimeout(Exception): pass

@contextmanager
def time_limit(seconds: float):
    def _handler(signum, frame): raise SympyTimeout()
    old = signal.signal(signal.SIGALRM, _handler)
    # seconds can be fractional via ITIMER_REAL
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def _cheap_norm(expr):
    # very fast, avoids trig/FU pipeline
    expr = powdenest(expr, force=True)
    expr = powsimp(expr, force=True, combine='exp')
    # ratsimp is relatively cheap and cancels rational garbage
    expr = ratsimp(expr)
    return expr


def custom_sin(expr, term):
    return sin(expr)

def custom_cos(expr, term):
    return cos(expr)

def custom_exp(expr, term):
    return exp(expr)

def custom_log(expr, term):
    return log(expr)

def custom_identity(expr, term):
    return expr

def custom_expand(expr, term):
    return expand(expr)

def custom_simplify(expr, term):
    return simplify(expr)

def custom_factor(expr, term):
    return factor(expr)

def custom_collect(expr, term):
    return collect(expr, term)

def custom_together(expr, term):
    return together(expr)

def custom_ratsimp(expr, term):
    return ratsimp(expr)

def custom_square(expr, term):
    return expr**2


def _cheap_norm(expr):
    # very fast, avoids trig/FU pipeline
    expr = powdenest(expr, force=True)
    expr = powsimp(expr, force=True, combine='exp')
    # ratsimp is relatively cheap and cancels rational garbage
    expr = ratsimp(expr)
    return expr

# def custom_sqrt(expr, term=None):
#     # apply sqrt without global simplify; only do local cheap normalization
#     expr = _cheap_norm(expr)
#     return sqrt(expr)


def custom_sqrt(expr, term=None, *, ops_limit: int = OPS_SIMPLIFY_LIMIT):
    """
    Fast & safe sqrt:
      1) If expr is structurally z**2, return z.
      2) Try powdenest (cheap) to expose hidden squares.
      3) Only if small enough (by count_ops), try simplify to see if it becomes z**2.
      4) Otherwise return sqrt(expr). Never raise.
    """
    try:
        # 1) trivial structural square
        if isinstance(expr, Pow) and expr.exp == 2:
            return expr.base

        # 2) cheap denesting (no trig simpl)
        expr_den = powdenest(expr, force=True)
        if isinstance(expr_den, Pow) and expr_den.exp == 2:
            return expr_den.base

        # 3) guard heavy simplify behind a size check
        if count_ops(expr_den) <= ops_limit:
            simplified = simplify(expr_den)
            if isinstance(simplified, Pow) and simplified.exp == 2:
                return simplified.base

        # 4) default principal branch
        return sym_sqrt(expr)
    except Exception:
        # any symbolic hiccup → no-op to keep the env stable
        return expr


def custom_sqrt_old(expr, term):
    # Check if the expression is a perfect square
    simplified_expr = simplify(expr)

    # Case 1: If it's a square of a single term (like x**2), return the term
    if simplified_expr.is_Pow and simplified_expr.exp == 2:
        base = simplified_expr.base
        return base

    # Case 2: Otherwise, return ±sqrt(expression)
    return sqrt(expr)

def inverse_sin(expr, term):
    if isinstance(expr, (int, float)):
        return asin(expr)
    if expr.has(sin):
        return expr.replace(
            lambda arg: arg.func == sin,
            lambda arg: arg.args[0]
        )
    return asin(expr)

def inverse_cos(expr, term):
    if isinstance(expr, (int, float)):
        return acos(expr)
    if expr.has(cos):
        return expr.replace(
            lambda arg: arg.func == cos,
            lambda arg: arg.args[0]
        )
    return acos(expr)

def inverse_tan(expr, term):
    if isinstance(expr, (int, float)):
        return atan(expr)
    if expr.has(tan):
        return expr.replace(
            lambda arg: arg.func == tan,
            lambda arg: arg.args[0]
        )
    return atan(expr)


from sympy import Add, symbols, simplify, sympify, S

def decompose_disjoint_blocks(expr, var):
    """
    Split expr into:
      residual_const: additive var-free part with any coefficients that also appear
                      multiplicatively removed (so blocks are disjoint)
      groups: [(dep_part, coeff)] with coefficients summed for identical dep_part
    """
    terms = expr.as_ordered_terms() if expr.is_Add else [expr]
    coeffs_by_dep, order_deps = {}, []
    const_add = S(0)

    for t in terms:
        const, dep = t.as_independent(var, as_Add=False)
        if dep == 1:
            const_add += const
        else:
            if dep not in coeffs_by_dep:
                order_deps.append(dep)
                coeffs_by_dep[dep] = 0
            coeffs_by_dep[dep] += const

    # Split const_add into individual terms if it's a sum
    if const_add.is_Add:
        const_terms = const_add.args
    else:
        const_terms = [const_add] if const_add != 0 else []

    residual_const = S(0)
    for ct in const_terms:
        residual_const += ct

    const_summands = set(const_terms)
    overlap = [c for c in coeffs_by_dep.values() if c in const_summands]
    residual_const -= (Add(*overlap) if overlap else 0)

    groups = [(dep, simplify(coeffs_by_dep[dep])) for dep in order_deps]
    return residual_const, groups

def relabel_with_existing_constants(lhs, rhs, var, terms=None, strategy="partial", include_expr_constants=True):
    """
    Relabel disjoint constant blocks across lhs and rhs using a fixed pool of labels,
    ensuring constants on opposite sides aren't reused or mapped inconsistently.

    Args:
      lhs, rhs: SymPy expressions representing left and right sides of the equation.
      var: The variable of interest (e.g., 'x').
      terms: List/tuple of Symbols to use as labels, in priority order.
             Default = [a, b, c] (created if not present).
      strategy: What to do if blocks > available labels:
          - "partial" (default): Replace as many blocks as possible, leave the rest unchanged.
          - "skip": Make no changes (return original expressions as-is).
          - "raise": Raise ValueError.
      include_expr_constants: If True, append constants from lhs and rhs (free_symbols minus {var})
                              AFTER the provided `terms` (deduplicated, order-stable).

    Returns:
      (new_lhs, new_rhs, mapping) where mapping is {original_block_expr -> chosen_symbol}
      for the blocks that were actually replaced across both sides.
    """
    if terms is None:
        a, b, c = symbols('a b c')
        pool = [a, b, c]
    else:
        pool = list(terms)

    lhs = sympify(lhs)
    rhs = sympify(rhs)

    # Collect all constants from both sides, excluding var
    all_constants = set()
    if include_expr_constants:
        lhs_constants = [s for s in sorted(lhs.free_symbols, key=lambda s: s.name) if s != var]
        rhs_constants = [s for s in sorted(rhs.free_symbols, key=lambda s: s.name) if s != var]
        all_constants.update(lhs_constants)
        all_constants.update(rhs_constants)
        for s in sorted(all_constants, key=lambda s: s.name):
            if s not in pool:
                pool.append(s)

    # Identify shared constants that appear independently on both sides
    shared_constants = set(lhs_constants) & set(rhs_constants)

    # Decompose both sides
    lhs_residual_const, lhs_groups = decompose_disjoint_blocks(lhs, var)
    rhs_residual_const, rhs_groups = decompose_disjoint_blocks(rhs, var)

    # Collect blocks, skipping shared independent constants
    blocks = []
    block_sides = []
    if lhs_residual_const != 0 and lhs_residual_const not in shared_constants:
        blocks.append(lhs_residual_const)
        block_sides.append('lhs')
    for _, coeff in lhs_groups:
        if coeff not in (0, 1) and coeff not in shared_constants:
            blocks.append(coeff)
            block_sides.append('lhs')
    if rhs_residual_const != 0 and rhs_residual_const not in shared_constants:
        blocks.append(rhs_residual_const)
        block_sides.append('rhs')
    for _, coeff in rhs_groups:
        if coeff not in (0, 1) and coeff not in shared_constants:
            blocks.append(coeff)
            block_sides.append('rhs')

    needed = len(blocks)
    available = len(pool)

    if needed == 0:
        return lhs, rhs, {}

    if available == 0:
        if strategy == "skip":
            return lhs, rhs, {}
        elif strategy == "partial":
            return lhs, rhs, {}
        else:
            raise ValueError("No labels available in pool.")

    if needed > available:
        if strategy == "skip":
            return lhs, rhs, {}
        elif strategy == "raise":
            raise ValueError(f"Need {needed} labels but only {available} available: {pool}")
        k = available
    else:
        k = needed

    # Build assignment, avoiding reuse across sides
    blk2sym = {}
    used_symbols = set()
    for i in range(k):
        blk = blocks[i]
        side = block_sides[i]
        # Determine constants on the other side
        other_side_constants = rhs.free_symbols if side == 'lhs' else lhs.free_symbols
        other_side_constants = [s for s in other_side_constants if s != var]

        for sym in pool:
            if sym not in used_symbols and sym not in other_side_constants and sym not in shared_constants:
                blk2sym[blk] = sym
                used_symbols.add(sym)
                break

    # Tentative assembly of new lhs and rhs
    new_lhs_terms = []
    if lhs_residual_const != 0:
        if lhs_residual_const in blk2sym:
            new_lhs_terms.append(blk2sym[lhs_residual_const])
        else:
            new_lhs_terms.append(lhs_residual_const)
    for dep, coeff in lhs_groups:
        if coeff == 0:
            continue
        if coeff == 1:
            new_lhs_terms.append(dep)
        else:
            if coeff in blk2sym:
                new_lhs_terms.append(blk2sym[coeff] * dep)
            else:
                new_lhs_terms.append(coeff * dep)

    new_rhs_terms = []
    if rhs_residual_const != 0:
        if rhs_residual_const in blk2sym:
            new_rhs_terms.append(blk2sym[rhs_residual_const])
        else:
            new_rhs_terms.append(rhs_residual_const)
    for dep, coeff in rhs_groups:
        if coeff == 0:
            continue
        if coeff == 1:
            new_rhs_terms.append(dep)
        else:
            if coeff in blk2sym:
                new_rhs_terms.append(blk2sym[coeff] * dep)
            else:
                new_rhs_terms.append(coeff * dep)

    new_lhs = Add(*new_lhs_terms) if new_lhs_terms else lhs
    new_rhs = Add(*new_rhs_terms) if new_rhs_terms else rhs

    # Simplification check: if new equation has the same or more symbols, revert
    original_symbols = len(lhs.free_symbols | rhs.free_symbols)
    new_symbols = len(new_lhs.free_symbols | new_rhs.free_symbols)
    if new_symbols >= original_symbols:
        return lhs, rhs, {}

    return new_lhs, new_rhs, {val:key for key,val in blk2sym.items()}

operation_names = {
    add: "add",
    sub: "subtract",
    mul: "multiply",
    truediv: "divide",
    custom_expand: "expand",
    custom_simplify: "simplify",
    custom_factor: "factor",
    custom_collect: "collect",
    custom_together: "together",
    custom_ratsimp: "ratsimp",
    custom_square: "square",
    custom_sqrt: "sqrt",
    inverse_sin: 'sin^{-1}',
    inverse_cos: 'cos^{-1}',
    inverse_tan: 'tan^{-1}',
    custom_identity: 'identity',
    custom_sin: "sin",
    custom_cos: "cos",
    inverse_sin: "sin^{-1}",
    inverse_cos: "cos^{-1}",
    inverse_tan: "tan^{-1}",
    custom_exp: "exp",
    custom_log: "log"
}