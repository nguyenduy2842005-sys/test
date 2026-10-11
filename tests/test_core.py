"""Kiểm chứng lõi tối ưu. Chạy:  python -m pytest -q tests   (hoặc  python tests/test_core.py)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opt_core import (  # noqa: E402
    BeamProblem, StructInput, make_family, optimize_beam, optimize_column, optimize_frame,
    column_buckling_fem, column_critical_load_closed, column_uniform_load,
    FrameFEM, TrussFEM, crosscheck_with_core,
)
from scipy.special import beta as beta_fn

E = 2.1e8


def test_uniform_column_matches_euler():
    L, I = 4.0, 1e-4
    for bc, k in [("pinned", 1.0), ("cantilever", 2.0), ("clamped", 0.5)]:
        N = 200
        P, _ = column_buckling_fem(np.full(N, L / N), np.full(N, E * I), bc)
        Pe = math.pi ** 2 * E * I / (k * L) ** 2
        assert abs(P / Pe - 1) < 1e-4, (bc, P, Pe)


def test_keller_ratio_four_thirds():
    fam = make_family("circle")
    r = optimize_column(5.0, fam, E, "pinned", V=5.0 * 1e-3, run_numeric=False)
    assert abs(r.ratio_closed - 4.0 / 3.0) < 1e-12          # Keller (1960), n = 2
    assert abs(r.ratio_fem / r.ratio_closed - 1) < 0.01     # FEM trên profile rời rạc


def test_column_cantilever_matches_closed():
    fam = make_family("rect", ratio=0.5)
    r = optimize_column(3.0, fam, E, "cantilever", V=3.0 * 2e-3, run_numeric=False)
    assert abs(r.ratio_fem / r.ratio_closed - 1) < 0.01


def test_column_numeric_oc_reaches_closed_form():
    fam = make_family("circle")
    r = optimize_column(5.0, fam, E, "pinned", V=5.0 * 1e-3, run_numeric=True, N_num=80, iters_num=300)
    assert r.P_numeric / r.P_closed > 0.97, (r.P_numeric, r.P_closed)
    assert r.P_numeric / r.P_closed < 1.03


def test_fast_frame_matches_core():
    fam = make_family("rect", ratio=0.5)
    xy = [(0, 0), (0, 4), (3, 5), (6, 4), (6, 0)]
    conn = [(0, 1), (1, 2), (2, 3), (3, 4)]
    m = FrameFEM(xy, conn, fam, E, [(0, True, True, False), (4, True, True, False)],
                 [(1, 15.0, 0.0, 0.0)], [0, -10.0, -10.0, 0])
    m._nodal_loads = [(1, 15.0, 0.0, 0.0)]
    A = np.array([0.02, 0.015, 0.015, 0.02])
    c = crosscheck_with_core(m, A)
    assert c["rel_diff"] < 1e-10, c


def test_beam_closed_form_vs_fem_oc():
    fam = make_family("rect", ratio=0.5)
    L = 8.0
    prob = BeamProblem(L=L, bc="ss", udls=[(0.0, 0.0, L)], point_loads=[(100.0, L / 2)], fam=fam,
                       delta_max=0.02, x_ctrl=L / 2, sigma_a=1e12, A_min=2e-4, n_elem=80)
    r = optimize_beam(prob)
    assert r.opt.feasible
    assert abs(r.vol_opt / r.vol_closed - 1) < 0.02, (r.vol_opt, r.vol_closed)
    assert r.crosscheck["rel_diff"] < 1e-9
    # E–L đại số: n=2, nén tải tập trung => M·m̄ = M²/P ⇒ A ∝ |M|^(2/3)
    A_cf = np.interp(r.x_mid, r.closed["x"], r.closed["A"])
    mask = (r.x_mid > 0.15 * L) & (r.x_mid < 0.85 * L)
    err = np.sqrt(np.mean(((r.A_opt - A_cf) / A_cf)[mask] ** 2))
    assert err < 0.06, err


def test_beam_saving_positive_indeterminate():
    fam = make_family("ibeam", h=0.4)
    L = 10.0
    prob = BeamProblem(L=L, bc="fixed_fixed", udls=[(30.0, 0.0, L)], point_loads=[], fam=fam,
                       delta_max=0.015, x_ctrl=L / 2, sigma_a=2.1e5, A_min=5e-4, n_elem=60)
    r = optimize_beam(prob)
    assert r.opt.feasible
    assert r.vol_opt < r.vol_uniform
    assert r.crosscheck["rel_diff"] < 1e-9


def test_truss_two_bar_exact():
    fam = make_family("tube", ratio=0.05)
    # hai thanh đối xứng, tải đứng P tại đỉnh: N = P/(2 sinθ)
    xy = [(0, 0), (2, 0), (1, 1)]
    conn = [(0, 2), (1, 2)]
    P = 100.0
    m = TrussFEM(xy, conn, fam, E, [(0, True, True, False), (1, True, True, False)], [(2, 0.0, -P, 0.0)])
    A = np.array([1e-3, 1e-3])
    U = m.solve(A)
    N, _ = m.forces(U, A)
    expect = -P / (2 * math.sin(math.pi / 4))
    assert np.allclose(N, expect, rtol=1e-9)


def test_ten_bar_truss_benchmark():
    """Dàn 10 thanh chuẩn (Schmit): 360 in, 100 kip, E=1e4 ksi, [σ]=±25 ksi, |δ|≤2 in, A≥0.1 in².
    Cực trị đã biết ≈ 5060.85 lb (ρ = 0.1 lb/in³; xác nhận độc lập bằng SLSQP với đạo hàm sai phân).
    Bộ tối ưu coi cận bền A_lb là hằng số mỗi vòng nên với dàn SIÊU TĨNH có thể dừng ở điểm dừng cục bộ
    (hiện ≈ +0.3%); test yêu cầu nằm trong +0.5% và không được thấp hơn cực trị (nếu thấp hơn ⇒ vi phạm ràng buộc)."""
    inch = 0.0254; kip = 4.4482216; ksi = 6894.757
    nodes = [(720, 360), (720, 0), (360, 360), (360, 0), (0, 360), (0, 0)]
    nodes = [(x * inch, y * inch) for x, y in nodes]
    members = [(2, 4), (0, 2), (3, 5), (1, 3), (2, 3), (0, 1), (3, 4), (2, 5), (1, 2), (0, 3)]
    sup = [(4, True, True, False), (5, True, True, False)]
    loads = [(1, 0.0, -100 * kip, 0.0), (3, 0.0, -100 * kip, 0.0)]
    lim = [(n, d, 2 * inch) for n in range(4) for d in ("ux", "uy")]
    fam = make_family("tube", ratio=0.05)
    inp = StructInput(nodes, members, sup, loads, [], "truss", fam, E=1e4 * ksi, sigma_a=25 * ksi,
                      rho=0.1 * 4.4482216 / 0.0254 ** 3, check_buckling=False, A_min=0.1 * inch ** 2,
                      A_max=1.0, disp_limits=lim)
    r = optimize_frame(inp)
    w_lb = r.weight / 4.4482216
    print("10-bar weight [lb] =", round(w_lb, 1), " iters", r.opt.iters, "feasible", r.opt.feasible)
    assert r.opt.feasible
    assert 4900 < w_lb < 5200, w_lb


def test_reactions_equilibrium_sign():
    """Phản lực phải CÂN BẰNG với tải ngoài (ΣR + ΣF = 0) — hồi quy cho lỗi dấu phản lực."""
    fam = make_family("tube", ratio=0.05)
    inp = StructInput([(0, 0), (6, 0), (0, 4), (6, 4)], [(0, 2), (2, 3), (1, 3)],
                      [(0, True, True, True), (1, True, True, True)],
                      [(2, 10.0, -20.0, 0.0)], [0.0, -15.0, 0.0], "frame", fam, E=E, sigma_a=2.1e5,
                      check_buckling=False, A_min=2e-4, disp_limits=[(2, "ux", 0.05)])
    r = optimize_frame(inp)
    m = r.model; dpn = 3; xy = m.xy
    rx = sum(v for d, v in r.reactions.items() if d % dpn == 0)
    ry = sum(v for d, v in r.reactions.items() if d % dpn == 1)
    assert abs(float(np.sum(m.F[0::3])) + rx) < 1e-8
    assert abs(float(np.sum(m.F[1::3])) + ry) < 1e-8
    Mext = float(np.sum(xy[:, 0] * m.F[1::3] - xy[:, 1] * m.F[0::3]) + np.sum(m.F[2::3]))
    Mr = sum((-xy[d // 3, 1] * v if d % 3 == 0 else xy[d // 3, 0] * v if d % 3 == 1 else v)
             for d, v in r.reactions.items())
    assert abs(Mext + Mr) < 1e-7


def test_column_oc_history_monotone_clamped():
    """Hồi quy: OC từng dao động mạnh (P/P* xuống 0.2) ở cột hai đầu ngàm; nay phải đơn điệu và ≥ 98%."""
    fam = make_family("tube", ratio=0.05)
    r = optimize_column(5.0, fam, E, "clamped", P_req=500.0, floor_frac=0.01, run_numeric=True)
    h = np.array(r.history)
    assert np.all(np.diff(h) >= -1e-9 * h[-1])
    assert r.P_numeric / r.P_closed > 0.98


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time(); fn(); print(f"OK  {name}  ({time.time() - t:.1f}s)")
