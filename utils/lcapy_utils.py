from copy import copy

import control
import lcapy as lca
import numpy as np
import sympy as sym
from IPython.display import Math, display
from lcapy.texpr import TimeDomainExpression
from scipy import signal

sym.init_printing()

default_assumptions = {
    "positive": True,
    "extended_nonpositive": False,
    "zero": False,
    "hermitian": True,
    "nonnegative": True,
    "negative": False,
    "infinite": False,
    "nonpositive": False,
    "extended_negative": False,
    "extended_positive": True,
    "extended_nonzero": True,
    "extended_real": True,
    "imaginary": False,
    "nonzero": True,
    "commutative": True,
    "extended_nonnegative": True,
    "real": True,
    "complex": True,
    "finite": True,
}


def display_lcapy_ss(ss: lca.StateSpace):
    eq_states = sym.latex(ss.dotx.expr.expand())
    eq_states += " &= " + sym.latex(sym.N(ss.A.expr.expand()))
    eq_states += " \, " + sym.latex(sym.N(ss.x.expr.expand()))
    eq_states += " + " + sym.latex(sym.N(ss.B.expr.expand()))
    eq_states += " \, " + sym.latex(sym.N(ss.u.expr.expand()))

    eq_outputs = sym.latex(ss.y.expr.expand())
    eq_outputs += " &= " + sym.latex(sym.N(ss.C.expr.expand()))
    eq_outputs += " \, " + sym.latex(sym.N(ss.x.expr.expand()))
    eq_outputs += " + " + sym.latex(sym.N(ss.D.expr.expand()))
    eq_outputs += " \, " + sym.latex(sym.N(ss.u.expr.expand()))

    latex = r"\begin{aligned}" + eq_states + r" \\ " + eq_outputs + r"\end{aligned}"

    display(Math(latex))


def matrix_eq_to_system_of_linear_eq(matrix_eq):
    return [
        sym.Eq(matrix_eq.lhs[r], matrix_eq.rhs[r]) for r in range(matrix_eq.lhs.rows)
    ]


def get_transfer_function(lin_eqs, num, den):
    H = sym.Symbol("H")
    _num = H * den
    result = []
    for eq in lin_eqs:
        tf = sym.solve(eq.subs({num: _num}), H)
        result.append(tf)
    result = filter(None, result)
    flatten = sum([*result], [])
    result = []
    for expr in flatten:
        result.append(sym.Eq(num / den, expr.expand().factor(deep=True)))
    return result


def solve_linear_system_of_equations(lin_eqs, Y, U):
    _vars = list(set([*Y.free_symbols, *U.free_symbols]))
    result = sym.nonlinsolve(lin_eqs, _vars)
    simpler = []
    for eq_set in result:
        for i, eq in enumerate(eq_set):
            simpler.append(
                sym.Eq(_vars[i], eq.expand().factor(deep=True), evaluate=False)
            )
    return simpler


def sympy_expr_to_function(expr, var=lca.t):
    return sym.Function(expr.name, **default_assumptions)(var)


def sympy_expr_to_symbol(expr):
    return sym.Symbol(expr.name, **default_assumptions)


def sympy_put_something_on(expr, something):
    # if expr.name[0] == '\\':
    #     return expr
    # s = expr.name.split('}')
    # name = f"\{something}" + "{" + "}".join([s[0], "", *s[1:]])
    name = f"\{something}" + "{" + expr.name[0] + "}" + expr.name[1:]
    if expr.is_Function:
        ret_expr = sym.Function(name, **default_assumptions)(*expr.args)
    if expr.is_Symbol:
        ret_expr = sym.Symbol(name, **default_assumptions)
    return ret_expr


def sympy_put_hat_on(expr):
    return sympy_put_something_on(expr, "hat")


def sympy_put_bar_on(expr):
    return sympy_put_something_on(expr, "bar")


def get_averaged_large_signal_model(
    ss_on: lca.StateSpace, ss_off: lca.StateSpace
) -> lca.StateSpace:
    dt = TimeDomainExpression("d(t)")

    A1 = ss_on.A
    B1 = ss_on.B
    C1 = ss_on.C
    D1 = ss_on.D
    A2 = ss_off.A
    B2 = ss_off.B
    C2 = ss_off.C
    D2 = ss_off.D

    A = (A1 * (dt) + A2 * (1 - dt)).expr
    B = (B1 * (dt) + B2 * (1 - dt)).expr
    C = (C1 * (dt) + C2 * (1 - dt)).expr
    D = (D1 * (dt) + D2 * (1 - dt)).expr

    return lca.StateSpace(A, B, C, D, ss_on.u, ss_on.y, ss_on.x)


def get_averaged_small_signal_model(
    ss_on: lca.StateSpace, ss_off: lca.StateSpace
) -> lca.StateSpace:
    dt = TimeDomainExpression("d(t)")

    A1 = ss_on.A
    B1 = ss_on.B
    C1 = ss_on.C
    D1 = ss_on.D
    A2 = ss_off.A
    B2 = ss_off.B
    C2 = ss_off.C
    D2 = ss_off.D

    u = copy(ss_on).u.applyfunc(sympy_expr_to_symbol).applyfunc(sympy_put_bar_on)
    x = copy(ss_on).x.applyfunc(sympy_expr_to_symbol).applyfunc(sympy_put_bar_on)

    Dt = sympy_put_bar_on(sympy_expr_to_symbol(dt))
    A = lca.Matrix(A1 * (Dt) + A2 * (1 - Dt))
    C = lca.Matrix(C1 * (Dt) + C2 * (1 - Dt))
    B = lca.Matrix([[B1 * (Dt) + B2 * (1 - Dt), (A1 - A2) * x + (B1 - B2) * u]])
    D = lca.Matrix([[D1 * (Dt) + D2 * (1 - Dt), (C1 - C2) * x + (D1 - D2) * u]])

    hu = lca.Matrix(
        [
            *copy(ss_on).u,
            dt,
        ]
    ).applyfunc(sympy_put_hat_on)
    hy = copy(ss_on).y.applyfunc(sympy_put_hat_on)
    hx = copy(ss_on).x.applyfunc(sympy_put_hat_on)

    return lca.StateSpace(A, B, C, D, hu, hy, hx)


def select_outputs(ss: lca.StateSpace, outputs: list) -> lca.StateSpace:
    outputs = set(outputs)
    outputs_to_delete = set(range(ss.y.rows - 1))
    [
        outputs_to_delete.remove(output)
        for output in outputs
        if output in outputs_to_delete
    ]

    for i, _ in sorted(enumerate(ss.y), reverse=True):
        if i in outputs_to_delete:
            ss.y.row_del(i)
            ss.C.row_del(i)
            ss.D.row_del(i)
    return ss


def lcapy_statespace_to_pythoncontrol(ss: lca.StateSpace) -> control.StateSpace:
    return control.StateSpace(
        np.array(ss.A.expr, dtype=float),
        np.array(ss.B.expr, dtype=float),
        np.array(ss.C.expr, dtype=float),
        np.array(ss.D.expr, dtype=float),
    )


def lcapy_statespace_to_scipy(ss: lca.StateSpace) -> control.StateSpace:
    return signal.StateSpace(
        np.array(ss.A.expr, dtype=float),
        np.array(ss.B.expr, dtype=float),
        np.array(ss.C.expr, dtype=float),
        np.array(ss.D.expr, dtype=float),
    )
