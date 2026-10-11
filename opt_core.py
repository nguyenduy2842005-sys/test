"""
opt_core.py — Lõi tối ưu hoá kết cấu thép cho "Đớp Tool"

Hai tầng:
  A. Phép tính biến phân (Calculus of Variations) — biến thiết kế là hàm liên tục A(x), I(x)
       * Cột chịu nén (Lagrange / Keller / Tadjbakhsh): phương trình Euler–Lagrange dạng ODE
         -> nghiệm đóng qua hàm Beta không đầy đủ.
       * Dầm chịu uốn (min thể tích, ràng buộc độ võng): F không chứa A' -> Euler–Lagrange
         suy biến thành phương trình ĐẠI SỐ:  A^(n+1) = λ·n·M·m̄ / (E·α).
  B. Tối ưu số trên lưới FEM (tiêu chuẩn tối ưu OC + Fully-Stressed-Design)
       * Phân tích nội lực/chuyển vị dùng lại ma trận phần tử của fem_core.py (không sửa core).
       * Dùng cho dầm siêu tĩnh, khung phẳng và dàn.

Đơn vị: kN, m, kN/m²  (E thép ≈ 2.1e8 kN/m², ứng suất 1 MPa = 1e3 kN/m²).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve
from scipy.optimize import brentq
from scipy.special import beta as beta_fn
from scipy.special import betaincinv

from fem_core import (
    FrameElement, FrameNode, FramePointLoad, FrameSupport, PlaneFrameInput,
    _beam_element_stiffness,
    _frame_element_stiffness_local,
    _frame_internal_force_at,
    _frame_udl_local_to_global,
    _rotation_matrix,
    solve_plane_frame,
)

STEEL_E = 2.1e8      # kN/m²
STEEL_RHO = 78.5     # kN/m³ (trọng lượng riêng)
STEEL_SIGMA = 2.1e5  # kN/m² (210 MPa)


# ══════════════════════════════════════════════════════════════
#  1. HỌ TIẾT DIỆN   I = α·A^n ,  W = β·A^(n-γ)   (độ cao c ∝ A^γ)
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SectionFamily:
    name: str
    n: float
    alpha: float
    gamma: float
    beta: float
    desc: str = ""

    def I(self, A):
        return self.alpha * np.power(A, self.n)

    def W(self, A):
        return self.beta * np.power(A, self.n - self.gamma)

    def dI_dA(self, A):
        return self.n * self.alpha * np.power(A, self.n - 1.0)

    def A_from_W(self, W):
        return np.power(np.maximum(W, 0.0) / self.beta, 1.0 / (self.n - self.gamma))

    def A_from_I(self, I):
        return np.power(np.maximum(I, 0.0) / self.alpha, 1.0 / self.n)


def make_family(kind: str, **kw) -> SectionFamily:
    """Tạo họ tiết diện.
    kind: 'rect' (ratio=b/h) | 'circle' | 'tube' (ratio=t/D) | 'ibeam' (h) | 'custom' (n, alpha, gamma, beta)
    """
    if kind == "rect":
        r = float(kw.get("ratio", 0.5))
        return SectionFamily("Chữ nhật đặc đồng dạng (b/h=%.2f)" % r, 2.0, 1.0 / (12.0 * r), 0.5,
                             1.0 / (6.0 * math.sqrt(r)),
                             "Mọi kích thước tỷ lệ với nhau: I ∝ A²")
    if kind == "circle":
        return SectionFamily("Tròn đặc", 2.0, 1.0 / (4.0 * math.pi), 0.5,
                             1.0 / (4.0 * math.sqrt(math.pi)), "I ∝ A²")
    if kind == "tube":
        r = float(kw.get("ratio", 0.05))
        return SectionFamily("Ống tròn mỏng đồng dạng (t/D=%.3f)" % r, 2.0,
                             1.0 / (8.0 * math.pi * r), 0.5,
                             1.0 / (4.0 * math.sqrt(math.pi * r)),
                             "Ống thành mỏng, t/D cố định: I ∝ A²")
    if kind == "ibeam":
        h = float(kw.get("h", 0.3))
        return SectionFamily("Chữ I cánh mỏng, h=%.3f m cố định" % h, 1.0, h * h / 4.0, 0.0, h / 2.0,
                             "A = diện tích hai cánh, bỏ qua bụng: I = A·h²/4, n = 1")
    if kind == "custom":
        n = float(kw["n"]); g = float(kw["gamma"])
        return SectionFamily("Tuỳ chỉnh (n=%.2f)" % n, n, float(kw["alpha"]), g, float(kw["beta"]), "")
    raise ValueError(f"Họ tiết diện không hợp lệ: {kind}")


# ══════════════════════════════════════════════════════════════
#  2. CỘT — EULER–LAGRANGE DẠNG ODE, NGHIỆM ĐÓNG
# ══════════════════════════════════════════════════════════════
#  Cực đại tải tới hạn P với thể tích V = ∫A dx cho trước, I = α·A^n.
#  Rayleigh: P = min ∫EI w''² dx / ∫w'² dx . Điều kiện dừng theo A:
#       n·E·α·A^(n-1)·w''² = λ   (đều)       (*)
#  Kết hợp M = EI·w'' = -P·w  ⇒  A ∝ |w|^(2/(n+1))  và  w'' = -κ·|w|^((1-n)/(n+1))·sign(w).
#  Tích phân lần đầu + đổi biến u = A/Amax cho  ξ = I_u((n+1)/2, 1/2)  (Beta không đầy đủ chuẩn hoá)
#  ⇒  u(ξ) = betaincinv((n+1)/2, 1/2, ξ).
#  Hệ quả:  <A>/Amax = (n+1)/(n+2) ;   P = (n+1)·B((n+1)/2,1/2)²·E·α·Amax^n / L_ef².

COLUMN_BCS = {
    "pinned": "Hai đầu khớp",
    "cantilever": "Một đầu ngàm (x=0) – một đầu tự do (x=L)",
    "clamped": "Hai đầu ngàm",
}


def _column_xi(bc: str, t: np.ndarray) -> np.ndarray:
    """ξ ∈ [0,1] (0: tiết diện triệt tiêu, 1: tiết diện lớn nhất) theo toạ độ chuẩn hoá t=x/L."""
    t = np.asarray(t, float)
    if bc == "pinned":
        return 1.0 - np.abs(1.0 - 2.0 * t)
    if bc == "cantilever":
        return 1.0 - t                       # ngàm ở x=0 (A max), đầu tự do ở x=L (A→0)
    if bc == "clamped":
        tt = np.minimum(t, 1.0 - t)
        return np.abs(1.0 - 4.0 * tt)        # A→0 tại L/4 và 3L/4
    raise ValueError(bc)


def _column_len_factor(bc: str) -> float:
    """Chiều dài hiệu dụng / L."""
    return {"pinned": 1.0, "cantilever": 2.0, "clamped": 0.5}[bc]


def column_profile(n: float, bc: str, L: float, V: float, x: np.ndarray) -> np.ndarray:
    """A(x) tối ưu dạng đóng (có thể bằng 0 ở các điểm đặc biệt)."""
    a = (n + 1.0) / 2.0
    Amax = V / (L * (n + 1.0) / (n + 2.0))
    xi = np.clip(_column_xi(bc, np.asarray(x) / L), 0.0, 1.0)
    return Amax * betaincinv(a, 0.5, xi)


def column_critical_load_closed(n: float, alpha: float, E: float, L: float, V: float, bc: str) -> float:
    Amax = V / (L * (n + 1.0) / (n + 2.0))
    B = beta_fn((n + 1.0) / 2.0, 0.5)
    Lef = _column_len_factor(bc) * L
    return (n + 1.0) * B ** 2 * E * alpha * Amax ** n / Lef ** 2


def column_uniform_load(n: float, alpha: float, E: float, L: float, V: float, bc: str) -> float:
    A0 = V / L
    Lef = _column_len_factor(bc) * L
    return math.pi ** 2 * E * alpha * A0 ** n / Lef ** 2


def _geo_stiffness(Le: float) -> np.ndarray:
    return 1.0 / (30.0 * Le) * np.array([
        [36, 3 * Le, -36, 3 * Le],
        [3 * Le, 4 * Le ** 2, -3 * Le, -Le ** 2],
        [-36, -3 * Le, 36, -3 * Le],
        [3 * Le, -Le ** 2, -3 * Le, 4 * Le ** 2],
    ])


def column_buckling_fem(L_el: np.ndarray, EI_el: np.ndarray, bc: str):
    """Tải tới hạn bằng FEM (ma trận độ cứng Hermite của fem_core + ma trận độ cứng hình học).
    Gốc x=0 là đầu ngàm với 'cantilever'. Trả (P_cr, vector chuyển vị đầy đủ)."""
    N = len(EI_el)
    nd = 2 * (N + 1)
    K = np.zeros((nd, nd)); G = np.zeros((nd, nd))
    for e in range(N):
        d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(d, d)] += _beam_element_stiffness(float(EI_el[e]), float(L_el[e]))
        G[np.ix_(d, d)] += _geo_stiffness(float(L_el[e]))
    if bc == "pinned":
        fixed = [0, 2 * N]
    elif bc == "cantilever":
        fixed = [0, 1]
    elif bc == "clamped":
        fixed = [0, 1, 2 * N, 2 * N + 1]
    else:
        raise ValueError(bc)
    free = [d for d in range(nd) if d not in fixed]
    w, V = eigh(K[np.ix_(free, free)], G[np.ix_(free, free)], subset_by_index=[0, 0])
    U = np.zeros(nd); U[free] = V[:, 0]
    return float(w[0]), U


@dataclass
class ColumnResult:
    bc: str
    L: float
    V: float
    fam: SectionFamily
    E: float
    x: np.ndarray
    A_opt: np.ndarray            # nghiệm E–L dạng đóng (sau khi áp cận dưới nếu có)
    A_uniform: float
    A_numeric: np.ndarray | None
    x_numeric: np.ndarray | None
    P_closed: float              # công thức đóng (không có cận dưới)
    P_fem_opt: float             # FEM cho profile đã dùng
    P_uniform: float
    P_numeric: float | None
    ratio_closed: float
    ratio_fem: float
    floor_frac: float
    history: list[float]
    report: str = ""


def optimize_column(L: float, fam: SectionFamily, E: float, bc: str,
                    V: float | None = None, P_req: float | None = None,
                    floor_frac: float = 0.0, N_check: int = 400,
                    run_numeric: bool = True, N_num: int = 80, iters_num: int = 250) -> ColumnResult:
    """Tối ưu cột. Cho V → tối đa P_cr; hoặc cho P_req → tối thiểu V (cùng nghiệm hình dạng)."""
    n, alpha = fam.n, fam.alpha
    if (V is None) == (P_req is None):
        raise ValueError("Chỉ chọn một trong V hoặc P_req.")
    if P_req is not None:
        # P ∝ V^n  →  V = V1 · (P_req/P1)^(1/n)
        V1 = L * 1e-3
        P1 = column_critical_load_closed(n, alpha, E, L, V1, bc)
        V = V1 * (P_req / P1) ** (1.0 / n)
    B = beta_fn((n + 1.0) / 2.0, 0.5)

    x = np.linspace(0.0, L, 1201)
    A_cf = column_profile(n, bc, L, V, x)

    P_closed = column_critical_load_closed(n, alpha, E, L, V, bc)
    P_uni = column_uniform_load(n, alpha, E, L, V, bc)

    # FEM kiểm chứng trên chính profile đóng (lấy giá trị tại trung điểm phần tử)
    xe = np.linspace(0.0, L, N_check + 1)
    xm = 0.5 * (xe[:-1] + xe[1:])
    Aprof = column_profile(n, bc, L, V, xm)
    Amax = V / (L * (n + 1.0) / (n + 2.0))
    A_floor = max(floor_frac, 1e-4) * Amax
    Aprof = np.maximum(Aprof, A_floor)
    Aprof *= V / float(np.sum(Aprof * np.diff(xe)))
    P_fem, _ = column_buckling_fem(np.diff(xe), E * fam.I(Aprof), bc)

    # profile hiển thị = profile đã dùng cho FEM (nội suy theo x)
    A_disp = np.interp(x, xm, Aprof)

    A_num = x_num = None
    P_num = None
    hist: list[float] = []
    if run_numeric:
        x_num, A_num, P_num, hist = column_oc_numeric(L, V, fam, E, bc, N=N_num, iters=iters_num)

    res = ColumnResult(bc=bc, L=L, V=V, fam=fam, E=E, x=x, A_opt=A_disp, A_uniform=V / L,
                       A_numeric=A_num, x_numeric=x_num, P_closed=P_closed, P_fem_opt=P_fem,
                       P_uniform=P_uni, P_numeric=P_num, ratio_closed=P_closed / P_uni,
                       ratio_fem=P_fem / P_uni, floor_frac=floor_frac, history=hist)
    res.report = _column_report(res, B)
    return res


def column_oc_numeric(L: float, V: float, fam: SectionFamily, E: float, bc: str,
                      N: int = 80, iters: int = 250, eta: float = 0.5, floor: float = 1e-3):
    """Tối ưu số (OC): cập nhật A_e theo điều kiện dừng n·A^(n-1)·κ² = hằng, giữ nguyên thể tích."""
    n = fam.n
    xe = np.linspace(0.0, L, N + 1)
    Le = np.diff(xe)
    A = np.full(N, V / L)
    hist = []
    best = (-1.0, A.copy())
    for _ in range(iters):
        P, U = column_buckling_fem(Le, E * fam.I(A), bc)
        hist.append(P)
        if P > best[0]:
            best = (P, A.copy())
        th = U[1::2]
        kap = np.diff(th) / Le
        r = A ** (n - 1.0) * kap ** 2 + 1e-300
        rbar = float(np.sum(r * A * Le) / np.sum(A * Le))
        A = A * (r / rbar) ** (eta / (n + 1.0))
        A = np.maximum(A, floor * V / L)
        A *= V / float(np.sum(A * Le))
        if len(hist) > 30 and abs(hist[-1] - hist[-10]) < 1e-9 * hist[-1]:
            break
    P_best, A_best = best
    xm = 0.5 * (xe[:-1] + xe[1:])
    return xm, A_best, P_best, hist


def _column_report(r: ColumnResult, B: float) -> str:
    n = r.fam.n
    L = [
        "=" * 56,
        "   THUYẾT MINH — TỐI ƯU CỘT BẰNG PHÉP TÍNH BIẾN PHÂN",
        "=" * 56,
        "",
        "I. BÀI TOÁN",
        f"   Sơ đồ: {COLUMN_BCS[r.bc]},  L = {r.L:.3f} m,  E = {r.E:.4g} kN/m²",
        f"   Họ tiết diện: {r.fam.name}   (I = α·A^n, n = {n:g}, α = {r.fam.alpha:.5g})",
        f"   Thể tích (V = ∫A dx) = {r.V:.5g} m³",
        "   Mục tiêu: cực đại tải tới hạn P_cr  (tương đương cực tiểu V với P_cr cho trước).",
        "",
        "II. PHƯƠNG TRÌNH EULER–LAGRANGE",
        "1) Phiếm hàm Rayleigh: P = min_w ∫ E·α·A^n·w''² dx / ∫ w'² dx , ràng buộc ∫A dx = V.",
        "   Biến phân theo A (F không chứa A'):  n·E·α·A^(n-1)·w''² = λ  (đại số, đều trên toàn cột).",
        "2) Kết hợp M = EI·w'' = -P·w  ⇒  A ∝ |w|^(2/(n+1)),",
        "   và w'' = -κ·|w|^((1-n)/(n+1))·sign(w)  (ODE phi tuyến, tích phân lần đầu được).",
        "3) Đặt u = A/Amax:  ξ = I_u((n+1)/2, 1/2)  ⇒  u(ξ) = betaincinv((n+1)/2, 1/2, ξ).",
        f"   Trung bình A/Amax = (n+1)/(n+2) = {(n + 1) / (n + 2):.5f}",
        f"   Amax = V / (L·(n+1)/(n+2)) = {r.V / (r.L * (n + 1) / (n + 2)):.5g} m²",
        "",
        "III. KẾT QUẢ",
        f"   B((n+1)/2, 1/2) = {B:.6f}",
        f"   P_cr  (nghiệm đóng)        = {r.P_closed:.5g} kN",
        f"   P_cr  (FEM, profile tối ưu)= {r.P_fem_opt:.5g} kN"
        + (f"   [cận dưới A ≥ {r.floor_frac:.1%}·Amax]" if r.floor_frac > 0 else ""),
        f"   P_cr  (cột đều, cùng V)    = {r.P_uniform:.5g} kN",
        f"   Hệ số tăng tải: đóng = {r.ratio_closed:.4f} ;  FEM = {r.ratio_fem:.4f}",
    ]
    if r.P_numeric is not None:
        L.append(f"   Tối ưu số OC (FEM)         = {r.P_numeric:.5g} kN  (tỷ số {r.P_numeric / r.P_uniform:.4f})")
    L += [
        "",
        "IV. LƯU Ý",
        "   - Nghiệm E–L cho A → 0 tại các điểm mômen = 0 (hai đầu khớp, đầu tự do, 1/4–3/4 L khi hai đầu ngàm).",
        "     Thực tế phải áp cận dưới A_min (cắt, ổn định cục bộ, liên kết) → P_cr giảm nhẹ.",
        "   - Mô hình Euler–Bernoulli, vật liệu đàn hồi, chưa xét ổn định cục bộ/ứng suất chảy.",
    ]
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════
#  3. DẦM — EULER–LAGRANGE ĐẠI SỐ (dầm tĩnh định) + FEM-OC (mọi sơ đồ)
# ══════════════════════════════════════════════════════════════

BEAM_BCS = {
    "ss": "Dầm tựa đơn",
    "cantilever": "Dầm console (ngàm tại x=0)",
    "fixed_pinned": "Ngàm (x=0) – gối di động (x=L)",
    "fixed_fixed": "Hai đầu ngàm",
}


@dataclass
class BeamProblem:
    L: float
    bc: str
    udls: list[tuple[float, float, float]]          # (q hướng xuống +, x1, x2)
    point_loads: list[tuple[float, float]]          # (P hướng xuống +, x)
    fam: SectionFamily
    E: float = STEEL_E
    sigma_a: float = STEEL_SIGMA
    delta_max: float | None = None                  # m
    x_ctrl: float | None = None
    A_min: float = 1e-4
    A_max: float = 1.0
    n_elem: int = 100


def beam_statics(prob: BeamProblem, x: np.ndarray) -> np.ndarray:
    """Mômen uốn M(x) (căng thớ dưới > 0) cho dầm TĨNH ĐỊNH (ss hoặc cantilever)."""
    L = prob.L
    x = np.asarray(x, float)
    M = np.zeros_like(x)
    if prob.bc == "ss":
        RA = sum(P * (L - xp) / L for P, xp in prob.point_loads)
        RA += sum(q * (b - a) * (L - 0.5 * (a + b)) / L for q, a, b in prob.udls)
        M += RA * x
        for P, xp in prob.point_loads:
            M -= P * np.maximum(x - xp, 0.0)
        for q, a, b in prob.udls:
            inside = (x > a) & (x <= b)
            after = x > b
            M[inside] -= 0.5 * q * (x[inside] - a) ** 2
            M[after] -= q * (b - a) * (x[after] - 0.5 * (a + b))
        return M
    if prob.bc == "cantilever":
        for P, xp in prob.point_loads:
            M -= P * np.maximum(xp - x, 0.0)
        for q, a, b in prob.udls:
            lo = np.maximum(a, x)
            ln = np.maximum(b - lo, 0.0)
            M -= q * ln * (0.5 * (lo + b) - x)
        return M
    raise ValueError("Mômen giải tích chỉ có cho dầm tĩnh định.")


def beam_unit_moment(prob: BeamProblem, x: np.ndarray, xp: float) -> np.ndarray:
    """Mômen m̄(x) do lực đơn vị hướng xuống tại xp (cùng quy ước dấu với beam_statics)."""
    L = prob.L
    x = np.asarray(x, float)
    if prob.bc == "ss":
        return np.where(x <= xp, x * (L - xp) / L, xp * (L - x) / L)
    if prob.bc == "cantilever":
        return -np.maximum(xp - x, 0.0)
    raise ValueError


def beam_closed_form(prob: BeamProblem, m: int = 4001):
    """Nghiệm Euler–Lagrange ĐẠI SỐ:  A(x) = max(A_bền, A_min, c·g^(1/(n+1))),  g = max(M·m̄, 0).
    c xác định từ ràng buộc δ(xp) = δ_max (brentq)."""
    if prob.bc not in ("ss", "cantilever"):
        return None
    fam, E, L = prob.fam, prob.E, prob.L
    n, alpha = fam.n, fam.alpha
    x = np.linspace(0.0, L, m)
    M = beam_statics(prob, x)
    xp = prob.x_ctrl if prob.x_ctrl is not None else (L / 2 if prob.bc == "ss" else L)
    mb = beam_unit_moment(prob, x, xp)
    A_lb = np.maximum(fam.A_from_W(np.abs(M) / prob.sigma_a), prob.A_min)
    g = np.maximum(M * mb, 0.0)

    def A_of(c):
        return np.minimum(np.maximum(A_lb, c * g ** (1.0 / (n + 1.0))), prob.A_max)

    def delta(c):
        A = A_of(c)
        return float(np.trapezoid(M * mb / (E * fam.I(A)), x))

    if prob.delta_max is None or np.max(g) <= 0:
        return dict(x=x, A=A_lb, M=M, c=0.0, delta=delta(0.0), xp=xp, A_lb=A_lb)
    if delta(0.0) <= prob.delta_max:
        return dict(x=x, A=A_lb, M=M, c=0.0, delta=delta(0.0), xp=xp, A_lb=A_lb)
    lo, hi = 1e-6, 1e6
    f = lambda lc: delta(math.exp(lc)) - prob.delta_max
    try:
        lc = brentq(f, math.log(lo), math.log(hi), xtol=1e-12, maxiter=200)
    except ValueError:
        lc = math.log(hi)
    c = math.exp(lc)
    return dict(x=x, A=A_of(c), M=M, c=c, delta=delta(c), xp=xp, A_lb=A_lb)


# ══════════════════════════════════════════════════════════════
#  4. FEM TÁI DÙNG fem_core  (khung phẳng & dàn)
# ══════════════════════════════════════════════════════════════

class FrameFEM:
    """Khung phẳng 2D: nhúng ma trận phần tử / ma trận xoay / tải tương đương / nội lực của fem_core."""
    kind = "frame"
    dof_per_node = 3

    def __init__(self, xy, conn, fam: SectionFamily, E: float, supports, loads, q_local=None):
        self.xy = np.asarray(xy, float)
        self.conn = np.asarray(conn, int)
        self.fam = fam
        self.n_nodes = len(self.xy)
        self.n_el = len(self.conn)
        self.ndof = 3 * self.n_nodes
        self.E = float(E)
        self.q = np.zeros(self.n_el) if q_local is None else np.asarray(q_local, float)

        xi = self.xy[self.conn[:, 0]]; xj = self.xy[self.conn[:, 1]]
        d = xj - xi
        self.L = np.hypot(d[:, 0], d[:, 1])
        if np.any(self.L < 1e-12):
            raise ValueError("Có phần tử chiều dài = 0.")
        self.c = d[:, 0] / self.L; self.s = d[:, 1] / self.L
        self.T = [_rotation_matrix(self.c[e], self.s[e]) for e in range(self.n_el)]
        self.dofs = []
        for e in range(self.n_el):
            i, j = self.conn[e]
            self.dofs.append([3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2])
        self.k_ax = [_frame_element_stiffness_local(self.E, 1.0, 0.0, self.L[e]) for e in range(self.n_el)]
        self.k_bd = [_frame_element_stiffness_local(self.E, 0.0, 1.0, self.L[e]) for e in range(self.n_el)]

        self.F = np.zeros(self.ndof)
        for node, fx, fy, mz in loads:
            self.F[3 * node] += fx; self.F[3 * node + 1] += fy; self.F[3 * node + 2] += mz
        for e in range(self.n_el):
            if self.q[e] != 0.0:
                el = FrameElement(int(self.conn[e, 0]), int(self.conn[e, 1]), self.E, 1.0, 1.0, float(self.q[e]))
                self.F[self.dofs[e]] += _frame_udl_local_to_global(el, self.L[e], self.T[e])

        con = set()
        self.supports = list(supports)
        for node, ux, uy, rz in self.supports:
            if ux: con.add(3 * node)
            if uy: con.add(3 * node + 1)
            if rz: con.add(3 * node + 2)
        self.constrained = np.array(sorted(con), int)
        self.free = np.array([k for k in range(self.ndof) if k not in con], int)
        self._lu = None
        self._K = None

    # ---- giải
    def solve(self, A):
        A = np.asarray(A, float)
        I = self.fam.I(A)
        K = np.zeros((self.ndof, self.ndof))
        for e in range(self.n_el):
            kl = _frame_element_stiffness_local(self.E, A[e], I[e], self.L[e])
            kg = self.T[e].T @ kl @ self.T[e]
            d = self.dofs[e]
            K[np.ix_(d, d)] += kg
        Kff = K[np.ix_(self.free, self.free)]
        lu = lu_factor(Kff)
        U = np.zeros(self.ndof)
        U[self.free] = lu_solve(lu, self.F[self.free])
        self._lu, self._K = lu, K
        return U

    def unit_solve(self, dof: int):
        """Chuyển vị do tải đơn vị tại dof (dùng sau solve cùng A)."""
        rhs = np.zeros(self.ndof); rhs[dof] = 1.0
        Uu = np.zeros(self.ndof)
        Uu[self.free] = lu_solve(self._lu, rhs[self.free])
        return Uu

    def disp_sens(self, U, Uu, A):
        """q_e = Uuᵀ·(∂K/∂A_e)·U  (⇒ ∂δ/∂A_e = -q_e)."""
        A = np.asarray(A, float)
        dI = self.fam.dI_dA(A)
        out = np.zeros(self.n_el)
        for e in range(self.n_el):
            ul = self.T[e] @ U[self.dofs[e]]; uu = self.T[e] @ Uu[self.dofs[e]]
            out[e] = uu @ (self.k_ax[e] + dI[e] * self.k_bd[e]) @ ul
        return out

    def forces(self, U, A, stations=(0.0, 0.25, 0.5, 0.75, 1.0)):
        A = np.asarray(A, float)
        I = self.fam.I(A)
        N = np.zeros(self.n_el); M = np.zeros(self.n_el)
        for e in range(self.n_el):
            ul = self.T[e] @ U[self.dofs[e]]
            el = FrameElement(int(self.conn[e, 0]), int(self.conn[e, 1]), self.E, float(A[e]), float(I[e]),
                              float(self.q[e]))
            vals = [_frame_internal_force_at(el, ul, self.L[e], xi * self.L[e]) for xi in stations]
            N[e] = vals[0][0]
            M[e] = max(abs(v[2]) for v in vals)
        return N, M

    def element_diagram(self, U, A, npts=21):
        """(x_global, y_global, N, V, M) tại npts điểm dọc từng phần tử — để vẽ biểu đồ."""
        A = np.asarray(A, float); I = self.fam.I(A)
        out = []
        for e in range(self.n_el):
            ul = self.T[e] @ U[self.dofs[e]]
            el = FrameElement(int(self.conn[e, 0]), int(self.conn[e, 1]), self.E, float(A[e]), float(I[e]),
                              float(self.q[e]))
            xi = np.linspace(0, 1, npts)
            vals = np.array([_frame_internal_force_at(el, ul, self.L[e], t * self.L[e]) for t in xi])
            p0 = self.xy[self.conn[e, 0]]; p1 = self.xy[self.conn[e, 1]]
            out.append(dict(x=p0[0] + xi * (p1[0] - p0[0]), y=p0[1] + xi * (p1[1] - p0[1]),
                            N=vals[:, 0], V=vals[:, 1], M=vals[:, 2]))
        return out

    def reactions(self, U):
        R = self._K @ U - self.F          # K·U = F + R_gối  ⇒  R_gối = K·U − F  (đúng dấu: cân bằng ΣR + ΣF = 0)
        return {int(d): float(R[d]) for d in self.constrained}

    def node_disp(self, U):
        return U.reshape(self.n_nodes, 3)

    def check_stability(self, A0=1e-2):
        K = np.zeros((self.ndof, self.ndof))
        I = self.fam.I(np.full(self.n_el, A0))
        for e in range(self.n_el):
            kl = _frame_element_stiffness_local(self.E, A0, I[e], self.L[e])
            K[np.ix_(self.dofs[e], self.dofs[e])] += self.T[e].T @ kl @ self.T[e]
        w = np.linalg.eigvalsh(K[np.ix_(self.free, self.free)])
        if w[0] <= 1e-9 * w[-1]:
            raise ValueError("Kết cấu bất ổn định (cơ cấu) — kiểm tra lại liên kết/gối tựa.")


class TrussFEM:
    """Dàn phẳng liên kết khớp: 2 bậc tự do/nút, mỗi thanh chỉ chịu lực dọc."""
    kind = "truss"
    dof_per_node = 2

    def __init__(self, xy, conn, fam: SectionFamily, E: float, supports, loads, q_local=None):
        self.xy = np.asarray(xy, float)
        self.conn = np.asarray(conn, int)
        self.fam = fam
        self.n_nodes = len(self.xy)
        self.n_el = len(self.conn)
        self.ndof = 2 * self.n_nodes
        self.E = float(E)
        xi = self.xy[self.conn[:, 0]]; xj = self.xy[self.conn[:, 1]]
        d = xj - xi
        self.L = np.hypot(d[:, 0], d[:, 1])
        if np.any(self.L < 1e-12):
            raise ValueError("Có thanh chiều dài = 0.")
        self.c = d[:, 0] / self.L; self.s = d[:, 1] / self.L
        self.b = np.stack([-self.c, -self.s, self.c, self.s], axis=1)
        self.dofs = [[2 * i, 2 * i + 1, 2 * j, 2 * j + 1] for i, j in self.conn]
        self.F = np.zeros(self.ndof)
        for node, fx, fy, mz in loads:
            self.F[2 * node] += fx; self.F[2 * node + 1] += fy
        con = set()
        self.supports = list(supports)
        for node, ux, uy, rz in self.supports:
            if ux: con.add(2 * node)
            if uy: con.add(2 * node + 1)
        self.constrained = np.array(sorted(con), int)
        self.free = np.array([k for k in range(self.ndof) if k not in con], int)
        self.q = np.zeros(self.n_el)
        self._lu = None
        self._K = None

    def solve(self, A):
        A = np.asarray(A, float)
        K = np.zeros((self.ndof, self.ndof))
        for e in range(self.n_el):
            b = self.b[e]
            K[np.ix_(self.dofs[e], self.dofs[e])] += self.E * A[e] / self.L[e] * np.outer(b, b)
        lu = lu_factor(K[np.ix_(self.free, self.free)])
        U = np.zeros(self.ndof)
        U[self.free] = lu_solve(lu, self.F[self.free])
        self._lu, self._K = lu, K
        return U

    def unit_solve(self, dof: int):
        rhs = np.zeros(self.ndof); rhs[dof] = 1.0
        Uu = np.zeros(self.ndof)
        Uu[self.free] = lu_solve(self._lu, rhs[self.free])
        return Uu

    def disp_sens(self, U, Uu, A):
        out = np.zeros(self.n_el)
        for e in range(self.n_el):
            d = self.dofs[e]
            out[e] = self.E / self.L[e] * float(self.b[e] @ Uu[d]) * float(self.b[e] @ U[d])
        return out

    def forces(self, U, A, stations=None):
        A = np.asarray(A, float)
        N = np.array([self.E * A[e] / self.L[e] * float(self.b[e] @ U[self.dofs[e]]) for e in range(self.n_el)])
        return N, np.zeros(self.n_el)

    def reactions(self, U):
        R = self._K @ U - self.F          # K·U = F + R_gối  ⇒  R_gối = K·U − F  (đúng dấu: cân bằng ΣR + ΣF = 0)
        return {int(d): float(R[d]) for d in self.constrained}

    def node_disp(self, U):
        return U.reshape(self.n_nodes, 2)

    def check_stability(self, A0=1e-2):
        K = np.zeros((self.ndof, self.ndof))
        for e in range(self.n_el):
            K[np.ix_(self.dofs[e], self.dofs[e])] += self.E * A0 / self.L[e] * np.outer(self.b[e], self.b[e])
        w = np.linalg.eigvalsh(K[np.ix_(self.free, self.free)])
        if w[0] <= 1e-9 * w[-1]:
            raise ValueError("Dàn bất ổn định (cơ cấu) — thiếu thanh hoặc gối tựa.")


# ══════════════════════════════════════════════════════════════
#  5. BỘ TỐI ƯU: FSD (bền/ổn định) + OC (chuyển vị)
# ══════════════════════════════════════════════════════════════
#  min  W(A) = ρ·Σ A_e·L_e
#  s.t. U_e(A) = |N|/(A·σ_N) + |M|/(W(A)·σ_a) ≤ 1     (bền + ổn định Euler khi nén)
#       |δ_c(A)| ≤ δ_c,max
#  Điều kiện tối ưu (Euler–Lagrange rời rạc): B_e = -(∂δ/∂A_e)/L_e = Λ  đều cho các phần tử
#  không chạm cận (cận bền / A_min / A_max).

@dataclass
class OptSettings:
    fam: SectionFamily
    E: float = STEEL_E
    sigma_a: float = STEEL_SIGMA
    rho: float = STEEL_RHO
    FS_buckling: float = 1.0
    check_buckling: bool = True
    A_min: float = 1e-4
    A_max: float = 1.0
    disp_limits: list = field(default_factory=list)      # [(dof_toàn_cục, giới hạn>0)]
    L_buck: np.ndarray | None = None
    eta: float = 0.5
    relax: float = 0.7
    max_iter: int = 200
    tol: float = 1e-6
    p_active: float = 4.0


def utilization(A, N, M, st: OptSettings, Lb):
    A = np.asarray(A, float)
    I = st.fam.I(A); W = st.fam.W(A)
    sig_N = np.full_like(A, st.sigma_a)
    if st.check_buckling:
        sig_cr = (math.pi ** 2 * st.E * I / (A * np.asarray(Lb) ** 2)) / st.FS_buckling
        sig_N = np.where(N < 0, np.minimum(st.sigma_a, sig_cr), st.sigma_a)
    return np.abs(N) / (A * sig_N) + np.abs(M) / (W * st.sigma_a)


def required_area(N, M, st: OptSettings, Lb):
    n_el = len(N)
    lo = np.log(np.full(n_el, st.A_min)); hi = np.log(np.full(n_el, st.A_max))
    ok_lo = utilization(np.exp(lo), N, M, st, Lb) <= 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        feas = utilization(np.exp(mid), N, M, st, Lb) <= 1.0
        hi = np.where(feas, mid, hi)
        lo = np.where(feas, lo, mid)
    A = np.exp(hi)
    return np.where(ok_lo, st.A_min, A)


@dataclass
class OptResult:
    A: np.ndarray
    U: np.ndarray
    N: np.ndarray
    M: np.ndarray
    util: np.ndarray
    disp_ratio: np.ndarray
    weight: float
    weight_uniform: float
    A_uniform: float
    history: list[dict]
    converged: bool
    iters: int
    feasible: bool
    B: np.ndarray | None
    EL_cv: float | None
    L_el: np.ndarray


def _weight(A, model, st):
    return float(st.rho * np.sum(np.asarray(A) * model.L))


def _disp_grads(model, st, A, U):
    """Giá trị g_k = 1-|δ_k|/lim_k và ∂g_k/∂A_e (liên hợp). model.solve(A) phải vừa được gọi."""
    G = np.zeros((len(st.disp_limits), model.n_el)); g = np.zeros(len(st.disp_limits))
    for k, (dof, lim) in enumerate(st.disp_limits):
        Uu = model.unit_solve(dof)
        qe = model.disp_sens(U, Uu, A)
        sgn = 1.0 if U[dof] >= 0 else -1.0
        g[k] = 1.0 - abs(U[dof]) / lim
        G[k] = sgn * qe / lim
    return g, G


def _kkt_polish(model, st, A, Lb, evaluate, hist):
    """Vòng ngoài: cố định cận bền A_lb (FSD); vòng trong: min W s.t. ràng buộc chuyển vị, A_lb ≤ A ≤ A_max."""
    from scipy.optimize import minimize
    ne = model.n_el
    W0 = max(_weight(A, model, st), 1e-30)
    A_best, W_best = None, np.inf
    for outer in range(20):
        U, N, M, u, r = evaluate(A)
        A_lb = np.clip(required_area(N, M, st, Lb), st.A_min, st.A_max)
        y0 = np.log(np.clip(np.maximum(A, A_lb), st.A_min, st.A_max))
        cache = {}

        def ev(y):
            key = y.tobytes()
            if key not in cache:
                Ay = np.exp(y)
                Uy = model.solve(Ay)
                cache.clear(); cache[key] = (Ay, Uy)
            return cache[key]

        def f(y): return float(st.rho * np.sum(np.exp(y) * model.L) / W0)
        def fg(y): return st.rho * np.exp(y) * model.L / W0
        def cons(y):
            Ay, Uy = ev(y)
            return np.array([1.0 - abs(Uy[d]) / lim for d, lim in st.disp_limits])
        def cjac(y):
            Ay, Uy = ev(y); model.solve(Ay)
            _, G = _disp_grads(model, st, Ay, Uy)
            return G * Ay[None, :]

        sol = minimize(f, y0, jac=fg, method="SLSQP",
                       bounds=list(zip(np.log(A_lb), np.full(ne, math.log(st.A_max)))),
                       constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                       options=dict(maxiter=500, ftol=1e-14))
        A_new = np.clip(np.exp(sol.x), st.A_min, st.A_max)
        Un, Nn, Mn, un, rn = evaluate(A_new)
        feas = (np.max(un) <= 1.0 + 1e-6) and (rn.size == 0 or np.max(rn) <= 1.0 + 1e-6)
        Wn = _weight(A_new, model, st)
        hist.append(dict(iter=len(hist) + 1, weight=Wn, max_util=float(np.max(un)),
                         max_disp_ratio=float(np.max(rn)) if rn.size else 0.0,
                         dA=float(np.max(np.abs(np.log(A_new / A))))))
        if feas and Wn < W_best:
            A_best, W_best = A_new.copy(), Wn
        dA = float(np.max(np.abs(np.log(A_new / A))))
        A = A_new
        if dA < 1e-4:
            break
    if A_best is None:
        return A, None
    evaluate(A_best)
    return A_best, None


def _kkt_residual(model, st, A, U, Lb, N, M):
    """‖∇W − Σλ_k ∇δ_k‖ / ‖∇W‖ trên các phần tử tự do, λ ≥ 0 giải bằng NNLS (điều kiện Euler–Lagrange rời rạc)."""
    from scipy.optimize import nnls
    model.solve(A)
    _, G = _disp_grads(model, st, A, U)
    A_lb = required_area(N, M, st, Lb)
    free = (A > 1.02 * A_lb) & (A > 1.02 * st.A_min) & (A < 0.98 * st.A_max)
    if np.sum(free) < 2:
        return None
    gradW = st.rho * model.L
    act = np.array([abs(U[d]) / lim > 0.5 for d, lim in st.disp_limits])
    if not act.any():
        return None
    Gm = (G[act][:, free]).T            # KKT: ∇W = Σ λ_k ∇g_k  với g_k = 1-|δ_k|/lim ≥ 0, λ_k ≥ 0
    lam, res = nnls(Gm * A[free][:, None], gradW[free] * A[free])
    return float(res / np.linalg.norm(gradW[free] * A[free]))


def optimize_structure(model, st: OptSettings, A_init: np.ndarray | None = None) -> OptResult:
    ne = model.n_el
    Lb = st.L_buck if st.L_buck is not None else np.ones(ne)
    model.check_stability()

    def evaluate(A):
        U = model.solve(A)
        N, M = model.forces(U, A)
        u = utilization(A, N, M, st, Lb)
        r = np.array([abs(U[d]) / lim for d, lim in st.disp_limits]) if st.disp_limits else np.zeros(0)
        return U, N, M, u, r

    # --- thiết kế đều (một tiết diện cho mọi phần tử) = mốc so sánh
    def g_uniform(la):
        A = np.full(ne, math.exp(la))
        _, _, _, u, r = evaluate(A)
        return max(float(np.max(u)), float(np.max(r)) if r.size else 0.0) - 1.0

    la_lo, la_hi = math.log(st.A_min), math.log(st.A_max)
    if g_uniform(la_lo) <= 0:
        A_u = st.A_min
    elif g_uniform(la_hi) > 0:
        A_u = st.A_max
    else:
        A_u = math.exp(brentq(g_uniform, la_lo, la_hi, xtol=1e-10, maxiter=200))
    W_uniform = _weight(np.full(ne, A_u), model, st)

    A = np.full(ne, A_u) if A_init is None else np.asarray(A_init, float).copy()
    hist = []
    B_last = None
    converged = False
    it = 0
    for it in range(1, st.max_iter + 1):
        U, N, M, u, r = evaluate(A)
        A_lb = required_area(N, M, st, Lb)
        A_new = A_lb.copy()
        if st.disp_limits:
            Bsum = np.zeros(ne)
            for (dof, lim), rc in zip(st.disp_limits, r):
                Uu = model.unit_solve(dof)
                qe = model.disp_sens(U, Uu, A)
                sgn = 1.0 if U[dof] >= 0 else -1.0
                w = max(rc, 1e-3) ** st.p_active
                Bsum += w * np.maximum(sgn * qe, 0.0) / lim
            B = Bsum / model.L
            Bmax = float(np.max(B)) if np.max(B) > 0 else 1.0
            B = np.maximum(B, 1e-9 * Bmax)
            Bbar = float(np.sum(B * A * model.L) / np.sum(A * model.L))
            A_trial = A * (B / Bbar) ** st.eta
            B_last = B

            def Afun(s):
                return np.clip(np.maximum(s * A_trial, A_lb), st.A_min, st.A_max)

            def g(ls):
                Ux = model.solve(Afun(math.exp(ls)))
                return max(abs(Ux[d]) / lim for d, lim in st.disp_limits) - 1.0

            lo_s, hi_s = math.log(1e-3), math.log(1e3)
            if g(lo_s) <= 0:
                s = math.exp(lo_s)
            elif g(hi_s) > 0:
                s = math.exp(hi_s)
            else:
                s = math.exp(brentq(g, lo_s, hi_s, xtol=1e-10, maxiter=100))
            A_new = Afun(s)
            model.solve(A)  # khôi phục factor cho vòng lặp kế (unit_solve dùng cache)
        A_next = np.exp((1 - st.relax) * np.log(A) + st.relax * np.log(A_new))
        Wn = _weight(A_next, model, st)
        dA = float(np.max(np.abs(np.log(A_next / A))))
        hist.append(dict(iter=it, weight=Wn, max_util=float(np.max(u)),
                         max_disp_ratio=float(np.max(r)) if r.size else 0.0, dA=dA))
        A = A_next
        if len(hist) > 2 and dA < 1e-4 and abs(hist[-1]["weight"] - hist[-2]["weight"]) < st.tol * Wn:
            converged = True
            break

    # --- hoàn thiện KKT: nhân tử Lagrange thật cho các ràng buộc chuyển vị (SLSQP + đạo hàm liên hợp)
    EL_res = None
    if st.disp_limits:
        A, EL_res = _kkt_polish(model, st, A, Lb, evaluate, hist)

    # --- hiệu chỉnh cuối để đảm bảo thoả mãn ràng buộc
    for _ in range(80):
        U, N, M, u, r = evaluate(A)
        rmax = float(np.max(r)) if r.size else 0.0
        if np.max(u) <= 1.0 + 1e-9 and rmax <= 1.0 + 1e-9:
            break
        A = np.where(u > 1.0, A * u, A)
        if rmax > 1.0 + 1e-9:
            A = A * rmax ** 0.7
        A = np.clip(A, st.A_min, st.A_max)
    U, N, M, u, r = evaluate(A)
    feasible = bool(np.max(u) <= 1.0 + 1e-6 and (r.size == 0 or np.max(r) <= 1.0 + 1e-6))

    # --- sai số điều kiện E–L rời rạc (KKT) trên các phần tử "tự do" (không chạm cận)
    EL_cv = None
    if st.disp_limits:
        EL_cv = _kkt_residual(model, st, A, U, Lb, N, M)
    return OptResult(A=A, U=U, N=N, M=M, util=u, disp_ratio=r, weight=_weight(A, model, st),
                     weight_uniform=W_uniform, A_uniform=A_u, history=hist, converged=converged, iters=it,
                     feasible=feasible, B=B_last, EL_cv=EL_cv, L_el=model.L.copy())


# ══════════════════════════════════════════════════════════════
#  6. DẦM — ĐƯA BÀI TOÁN DẦM VỀ FEM + SO SÁNH NGHIỆM ĐÓNG
# ══════════════════════════════════════════════════════════════

@dataclass
class BeamOptResult:
    prob: BeamProblem
    xs: np.ndarray                    # toạ độ nút
    x_mid: np.ndarray
    A_opt: np.ndarray
    A_uniform: float
    I_opt: np.ndarray
    opt: OptResult
    closed: dict | None
    deflection_opt: np.ndarray        # võng tại nút (m, hướng xuống +)
    deflection_uni: np.ndarray
    moment_x: np.ndarray
    moment_opt: np.ndarray
    vol_opt: float
    vol_uniform: float
    vol_closed: float | None
    delta_ctrl_opt: float | None
    delta_ctrl_uni: float | None
    ctrl_node: int
    crosscheck: dict
    report: str = ""


def _beam_mesh(prob: BeamProblem):
    L = prob.L
    pts = list(np.linspace(0.0, L, prob.n_elem + 1))
    pts += [x for _, x in prob.point_loads]
    for _, a, b in prob.udls:
        pts += [a, b]
    xp = prob.x_ctrl if prob.x_ctrl is not None else (L / 2 if prob.bc != "cantilever" else L)
    pts.append(xp)
    pts = np.array(sorted(pts))
    keep = [pts[0]]
    for p in pts[1:]:
        if p - keep[-1] > 1e-9 * L:
            keep.append(p)
    return np.array(keep), xp


def build_beam_model(prob: BeamProblem):
    xs, xp = _beam_mesh(prob)
    nn = len(xs); ne = nn - 1
    xy = np.stack([xs, np.zeros(nn)], axis=1)
    conn = np.stack([np.arange(ne), np.arange(1, ne + 1)], axis=1)
    sup = {"ss": [(0, True, True, False), (nn - 1, False, True, False)],
           "cantilever": [(0, True, True, True)],
           "fixed_pinned": [(0, True, True, True), (nn - 1, False, True, False)],
           "fixed_fixed": [(0, True, True, True), (nn - 1, False, True, True)]}[prob.bc]
    loads = []
    for P, x in prob.point_loads:
        k = int(np.argmin(np.abs(xs - x)))
        loads.append((k, 0.0, -P, 0.0))
    q_local = np.zeros(ne)
    xm = 0.5 * (xs[:-1] + xs[1:])
    for q, a, b in prob.udls:
        q_local[(xm > a) & (xm < b)] += -q          # hướng xuống ⇒ trục y cục bộ âm
    model = FrameFEM(xy, conn, prob.fam, prob.E, sup, loads, q_local)
    model._nodal_loads = list(loads)
    ctrl = int(np.argmin(np.abs(xs - xp)))
    return model, xs, ctrl


def optimize_beam(prob: BeamProblem) -> BeamOptResult:
    model, xs, ctrl = build_beam_model(prob)
    ne = model.n_el
    limits = []
    if prob.delta_max is not None:
        limits.append((3 * ctrl + 1, prob.delta_max))
    st = OptSettings(fam=prob.fam, E=prob.E, sigma_a=prob.sigma_a, check_buckling=False,
                     A_min=prob.A_min, A_max=prob.A_max, disp_limits=limits, max_iter=300, relax=0.8)
    opt = optimize_structure(model, st)

    # chuyển vị của thiết kế đều (cùng ràng buộc)
    A_u = np.full(ne, opt.A_uniform)
    U_u = model.solve(A_u)
    U_o = model.solve(opt.A)
    model.solve(opt.A)

    # mômen theo trục x (tại nút) từ phần tử: lấy M tại hai đầu mỗi phần tử
    diag = model.element_diagram(U_o, opt.A, npts=3)
    mx = []; my = []
    for e, d in enumerate(diag):
        mx += [d["x"][0], d["x"][1]]
        my += [d["M"][0], d["M"][1]]
    mx.append(diag[-1]["x"][-1]); my.append(diag[-1]["M"][-1])

    closed = beam_closed_form(prob)
    x_mid = 0.5 * (xs[:-1] + xs[1:])
    vol_o = float(np.sum(opt.A * model.L))
    vol_u = float(opt.A_uniform * prob.L)
    vol_c = float(np.trapezoid(closed["A"], closed["x"])) if closed is not None else None
    d_o = d_u = None
    if limits:
        d_o = abs(float(U_o[3 * ctrl + 1])); d_u = abs(float(U_u[3 * ctrl + 1]))

    cross = crosscheck_with_core(model, opt.A)
    res = BeamOptResult(prob=prob, xs=xs, x_mid=x_mid, A_opt=opt.A, A_uniform=opt.A_uniform,
                        I_opt=prob.fam.I(opt.A), opt=opt, closed=closed,
                        deflection_opt=-U_o[1::3], deflection_uni=-U_u[1::3],
                        moment_x=np.array(mx), moment_opt=np.array(my),
                        vol_opt=vol_o, vol_uniform=vol_u, vol_closed=vol_c,
                        delta_ctrl_opt=d_o, delta_ctrl_uni=d_u, ctrl_node=ctrl, crosscheck=cross)
    res.report = _beam_report(res)
    return res


def crosscheck_with_core(model: FrameFEM, A) -> dict:
    """Đối chiếu bộ giải nhanh với fem_core.solve_plane_frame (nguyên bản)."""
    A = np.asarray(A, float)
    I = model.fam.I(A)
    nodes = [FrameNode(float(p[0]), float(p[1])) for p in model.xy]
    elems = [FrameElement(int(model.conn[e, 0]), int(model.conn[e, 1]), model.E, float(A[e]), float(I[e]),
                          float(model.q[e])) for e in range(model.n_el)]
    sups = [FrameSupport(node, bool(ux), bool(uy), bool(rz)) for node, ux, uy, rz in model.supports]
    # tải nút gốc (không gồm tải tương đương của UDL — core tự tính từ udl_local)
    pls = [FramePointLoad(node, fx, fy, mz) for node, fx, fy, mz in getattr(model, "_nodal_loads", [])]
    ref = solve_plane_frame(PlaneFrameInput(nodes, elems, sups, pls))
    U = model.solve(A)
    U_ref = ref.node_displacements.reshape(-1)
    dmax = float(np.max(np.abs(U - U_ref)))
    scale = float(np.max(np.abs(U_ref))) + 1e-300
    return dict(max_abs_diff=dmax, rel_diff=dmax / scale)


# ══════════════════════════════════════════════════════════════
#  7. KHUNG / DÀN — NHẬP LIỆU VÀ TỐI ƯU TOÀN KẾT CẤU
# ══════════════════════════════════════════════════════════════

@dataclass
class StructInput:
    nodes: list[tuple[float, float]]
    members: list[tuple[int, int]]
    supports: list[tuple[int, bool, bool, bool]]          # (node, ux, uy, rz)
    loads: list[tuple[int, float, float, float]]          # (node, Fx, Fy, Mz)
    member_q: list[float]                                 # q vuông góc thanh (khung)
    mode: str                                             # "frame" | "truss"
    fam: SectionFamily
    E: float = STEEL_E
    sigma_a: float = STEEL_SIGMA
    rho: float = STEEL_RHO
    FS_buckling: float = 1.0
    k_eff: float = 1.0
    check_buckling: bool = True
    A_min: float = 1e-4
    A_max: float = 1.0
    disp_limits: list[tuple[int, str, float]] = field(default_factory=list)   # (node, 'ux'|'uy', |δ|max m)
    segments: int = 1


@dataclass
class StructOptResult:
    inp: StructInput
    model: object
    member_of_el: np.ndarray
    opt: OptResult
    A_member: np.ndarray            # diện tích trung bình theo thanh gốc
    weight: float
    weight_uniform: float
    saving: float
    crosscheck: dict | None
    reactions: dict
    report: str = ""


def build_structure(inp: StructInput):
    nodes = [tuple(p) for p in inp.nodes]
    conn = []
    member_of_el = []
    q_el = []
    seg = max(1, int(inp.segments)) if inp.mode == "frame" else 1
    for m, (i, j) in enumerate(inp.members):
        if seg == 1:
            conn.append((i, j)); member_of_el.append(m)
            q_el.append(inp.member_q[m] if m < len(inp.member_q) else 0.0)
        else:
            pi, pj = np.array(nodes[i]), np.array(nodes[j])
            prev = i
            for k in range(1, seg):
                p = pi + (pj - pi) * k / seg
                nodes.append((float(p[0]), float(p[1])))
                cur = len(nodes) - 1
                conn.append((prev, cur)); member_of_el.append(m)
                q_el.append(inp.member_q[m] if m < len(inp.member_q) else 0.0)
                prev = cur
            conn.append((prev, j)); member_of_el.append(m)
            q_el.append(inp.member_q[m] if m < len(inp.member_q) else 0.0)
    cls = FrameFEM if inp.mode == "frame" else TrussFEM
    model = cls(nodes, conn, inp.fam, inp.E, inp.supports, inp.loads, q_el)
    model._nodal_loads = list(inp.loads)
    member_of_el = np.array(member_of_el, int)
    mem_len = np.array([math.hypot(inp.nodes[j][0] - inp.nodes[i][0], inp.nodes[j][1] - inp.nodes[i][1])
                        for i, j in inp.members])
    Lb = inp.k_eff * mem_len[member_of_el]
    dpn = model.dof_per_node
    limits = []
    for node, d, lim in inp.disp_limits:
        limits.append((dpn * node + (0 if d == "ux" else 1), float(lim)))
    st = OptSettings(fam=inp.fam, E=inp.E, sigma_a=inp.sigma_a, rho=inp.rho,
                     FS_buckling=inp.FS_buckling, check_buckling=inp.check_buckling,
                     A_min=inp.A_min, A_max=inp.A_max, disp_limits=limits, L_buck=Lb)
    return model, st, member_of_el


def optimize_frame(inp: StructInput) -> StructOptResult:
    if not inp.nodes or not inp.members:
        raise ValueError("Chưa có nút/thanh.")
    if not inp.supports:
        raise ValueError("Chưa khai báo gối tựa.")
    model, st, member_of_el = build_structure(inp)
    opt = optimize_structure(model, st)
    nm = len(inp.members)
    A_member = np.zeros(nm)
    for m in range(nm):
        sel = member_of_el == m
        A_member[m] = float(np.sum(opt.A[sel] * model.L[sel]) / np.sum(model.L[sel]))
    model.solve(opt.A)
    reac = model.reactions(opt.U)
    cross = None
    if inp.mode == "frame":
        cross = crosscheck_with_core(model, opt.A)
    res = StructOptResult(inp=inp, model=model, member_of_el=member_of_el, opt=opt, A_member=A_member,
                          weight=opt.weight, weight_uniform=opt.weight_uniform,
                          saving=1.0 - opt.weight / opt.weight_uniform if opt.weight_uniform > 0 else 0.0,
                          crosscheck=cross, reactions=reac)
    res.report = _struct_report(res)
    return res


# ══════════════════════════════════════════════════════════════
#  8. BÁO CÁO
# ══════════════════════════════════════════════════════════════

def _beam_report(r: BeamOptResult) -> str:
    p = r.prob; n = p.fam.n
    L = [
        "=" * 56,
        "   THUYẾT MINH — TỐI ƯU DẦM VÚT THON (BIẾN PHÂN + FEM)",
        "=" * 56,
        "",
        "I. BÀI TOÁN",
        f"   Sơ đồ: {BEAM_BCS[p.bc]},  L = {p.L:.3f} m,  E = {p.E:.4g} kN/m²,  [σ] = {p.sigma_a / 1e3:.1f} MPa",
        f"   Họ tiết diện: {p.fam.name}  (I = α·A^n, n = {n:g};  W = β·A^(n-γ), γ = {p.fam.gamma:g})",
        "   Tải trọng (hướng xuống dương):",
    ]
    for P, x in p.point_loads:
        L.append(f"     - P = {P:.3f} kN tại x = {x:.3f} m")
    for q, a, b in p.udls:
        L.append(f"     - q = {q:.3f} kN/m từ x = {a:.3f} đến {b:.3f} m")
    if p.delta_max is not None:
        xp = r.xs[r.ctrl_node]
        L.append(f"   Ràng buộc: |δ(x={xp:.3f} m)| ≤ {p.delta_max * 1e3:.3f} mm;  A ≥ {p.A_min:.3g} m²")
    else:
        L.append("   Ràng buộc: chỉ điều kiện bền (σ = |M|/W ≤ [σ])")
    L += [
        "",
        "II. PHƯƠNG TRÌNH EULER–LAGRANGE",
        "1) Phiếm hàm: J[A] = ∫A dx + λ·( ∫ M·m̄ /(E·α·A^n) dx - δ_max ).",
        "   F = A + λ·M·m̄/(E·α·A^n) KHÔNG chứa A' ⇒ F_A - d/dx(F_A') = 0 suy biến thành PT đại số:",
        "      1 - λ·n·M·m̄ /(E·α·A^(n+1)) = 0   ⇒   A(x) = [λ·n·M(x)·m̄(x)/(E·α)]^(1/(n+1))",
        f"   với n = {n:g}:  A ∝ |M·m̄|^(1/{n + 1:g}).",
        "2) Cận bền: A ≥ A_bền(x) = A_từ_W(|M|/[σ]); nghiệm KKT: A = max(A_bền, A_min, c·g^(1/(n+1))).",
        "3) Hằng số c xác định từ δ(c) = δ_max (giải bằng brentq).",
    ]
    if r.closed is not None:
        L.append(f"   c = {r.closed['c']:.5g}   |   δ(nghiệm đóng) = {r.closed['delta'] * 1e3:.4f} mm")
    else:
        L.append("   Dầm siêu tĩnh: M phụ thuộc A nên không có nghiệm đóng — dùng FEM + OC (chuẩn hoá B_e = hằng).")
    L += [
        "",
        "III. KẾT QUẢ",
        f"   Thể tích tối ưu (FEM-OC)    = {r.vol_opt:.5g} m³",
    ]
    if r.vol_closed is not None:
        L.append(f"   Thể tích nghiệm đóng (E–L)  = {r.vol_closed:.5g} m³  "
                 f"(lệch {abs(r.vol_opt - r.vol_closed) / r.vol_closed * 100:.2f}%)")
    L += [
        f"   Thể tích dầm đều (1 tiết diện) = {r.vol_uniform:.5g} m³   [A = {r.A_uniform:.5g} m²]",
        f"   Tiết kiệm vật liệu            = {(1 - r.vol_opt / r.vol_uniform) * 100:.2f}%",
        f"   A_max/A_min của thiết kế     = {np.max(r.A_opt):.5g} / {np.min(r.A_opt):.5g} m²",
    ]
    if r.delta_ctrl_opt is not None:
        L.append(f"   δ tại điểm khống chế: tối ưu = {r.delta_ctrl_opt * 1e3:.4f} mm ; dầm đều = "
                 f"{r.delta_ctrl_uni * 1e3:.4f} mm ; giới hạn = {p.delta_max * 1e3:.4f} mm")
    L += [
        f"   Hội tụ: {'có' if r.opt.converged else 'chưa'} sau {r.opt.iters} vòng lặp; "
        f"thoả ràng buộc: {'có' if r.opt.feasible else 'KHÔNG'}",
        f"   Đối chiếu bộ giải nhanh với fem_core.solve_plane_frame: sai khác chuyển vị tương đối "
        f"{r.crosscheck['rel_diff']:.2e}",
    ]
    if r.opt.EL_cv is not None:
        L.append(f"   Kiểm tra E–L rời rạc (KKT): sai số tương đối ∇W − Σλ∇δ trên đoạn tự do = {r.opt.EL_cv:.3%}")
    L += [
        "",
        "IV. LƯU Ý",
        "   - Mô hình Euler–Bernoulli, bỏ qua biến dạng cắt và ổn định cục bộ/tải cục bộ.",
        "   - Tiết diện thực phải làm tròn theo thép hình/tấm có sẵn và chế tạo vút thon liên tục (hoặc theo đoạn).",
    ]
    return "\n".join(L)


def _struct_report(r: StructOptResult) -> str:
    inp = r.inp; o = r.opt; m = r.model
    L = [
        "=" * 56,
        f"   THUYẾT MINH — TỐI ƯU {'KHUNG PHẲNG' if inp.mode == 'frame' else 'DÀN THÉP'} (FEM + OC/FSD)",
        "=" * 56,
        "",
        "I. BÀI TOÁN",
        f"   Số nút = {len(inp.nodes)},  số thanh = {len(inp.members)},  "
        f"liên kết nút = {'hàn cứng (khung)' if inp.mode == 'frame' else 'khớp (dàn)'}",
        f"   Họ tiết diện: {inp.fam.name}  (n = {inp.fam.n:g})",
        f"   E = {inp.E:.4g} kN/m²,  [σ] = {inp.sigma_a / 1e3:.1f} MPa,  ρg = {inp.rho:.1f} kN/m³",
        f"   Ổn định Euler khi nén: {'bật' if inp.check_buckling else 'tắt'}  (k = {inp.k_eff:g}, FS = {inp.FS_buckling:g})",
        f"   Cận diện tích: {inp.A_min:.3g} ≤ A ≤ {inp.A_max:.3g} m²" +
        (f";  số đoạn chia mỗi thanh = {inp.segments}" if inp.mode == "frame" else ""),
        "   Ràng buộc chuyển vị: " + (", ".join(f"|{d}(N{n})| ≤ {lim * 1e3:.2f} mm" for n, d, lim in inp.disp_limits)
                                      if inp.disp_limits else "không"),
        "",
        "II. PHƯƠNG PHÁP",
        "1) Phân tích: FEM khung/dàn (ma trận phần tử, ma trận xoay, tải tương đương lấy từ fem_core).",
        "2) Ràng buộc bền + ổn định:  U = |N|/(A·σ_N) + |M|/(W·[σ]) ≤ 1,  σ_N = min([σ], σ_Euler) khi nén.",
        "3) Ràng buộc chuyển vị: tối ưu theo tiêu chuẩn OC. Phiếm hàm Lagrange L = ρΣA·L + λ(δ - δ_max):",
        "      ∂L/∂A_e = 0  ⇔  B_e = -(∂δ/∂A_e)/L_e = Λ  (đều cho thanh không chạm cận).",
        "   Với dàn tĩnh định: A_e = √(λ·|N_e·n_e|/(E·ρ)) (điều kiện Schmit/Maxwell–Mohr).",
        "4) Vòng lặp: A_mới = max(A_bền, A_OC·s) với hệ số s giải từ δ = δ_max; hoà trộn lôgarit (relax).",
        "",
        "III. KẾT QUẢ",
        f"   Trọng lượng tối ưu   = {r.weight:.4f} kN",
        f"   Trọng lượng tiết diện đều = {r.weight_uniform:.4f} kN   (A đều = {o.A_uniform:.5g} m²)",
        f"   Tiết kiệm            = {r.saving * 100:.2f}%",
        f"   Hội tụ: {'có' if o.converged else 'chưa'} sau {o.iters} vòng; thoả ràng buộc: {'có' if o.feasible else 'KHÔNG'}",
        f"   Utilization lớn nhất = {float(np.max(o.util)):.4f}",
    ]
    if o.disp_ratio.size:
        L.append("   Tỷ số chuyển vị |δ|/giới hạn = " + ", ".join(f"{v:.4f}" for v in o.disp_ratio))
    if o.EL_cv is not None:
        L.append(f"   Kiểm tra E–L rời rạc (KKT): sai số tương đối ∇W − Σλ∇δ trên thanh tự do = {o.EL_cv:.3%}")
    if r.crosscheck is not None:
        L.append(f"   Đối chiếu fem_core.solve_plane_frame: sai khác chuyển vị tương đối = "
                 f"{r.crosscheck['rel_diff']:.2e}")
    L += ["", "1) Bảng thanh (giá trị trung bình theo thanh)"]
    nm = len(inp.members)
    L.append(f"   {'Thanh':>5} {'i-j':>7} {'L(m)':>8} {'A(cm²)':>10} {'N(kN)':>10} {'Mmax(kNm)':>10} {'U':>7}")
    for k in range(nm):
        sel = r.member_of_el == k
        Lk = float(np.sum(m.L[sel]))
        Nk = float(o.N[sel][np.argmax(np.abs(o.N[sel]))])
        Mk = float(np.max(o.M[sel]))
        Uk = float(np.max(o.util[sel]))
        i, j = inp.members[k]
        L.append(f"   {k:>5} {str(i) + '-' + str(j):>7} {Lk:>8.3f} {r.A_member[k] * 1e4:>10.3f} "
                 f"{Nk:>10.3f} {Mk:>10.3f} {Uk:>7.3f}")
    L += ["", "2) Phản lực gối"]
    names = ["Rx", "Ry", "Mz"] if inp.mode == "frame" else ["Rx", "Ry"]
    dpn = m.dof_per_node
    for d, v in sorted(r.reactions.items()):
        L.append(f"   Nút {d // dpn}: {names[d % dpn]} = {v:.4f}")
    L += ["", "IV. LƯU Ý",
          "   - FSD/OC cho nghiệm tốt thực hành; với kết cấu siêu tĩnh, nghiệm là cực trị địa phương.",
          "   - Chưa xét tải trọng bản thân, ổn định tổng thể bậc hai (P-Δ), mỏi và chi tiết liên kết.",
          "   - Tiết diện thực cần làm tròn theo thép hình/ống tiêu chuẩn rồi kiểm tra lại theo TCVN/AISC."]
    return "\n".join(L)
