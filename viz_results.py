"""
viz_results.py — lớp nối giữa kết quả tối ưu (opt_core) và các hàm vẽ của trang web cũ (legacy_plots).

Mục đích: hiển thị SƠ ĐỒ TẢI TRỌNG, LỰC CẮT, MÔMEN, CHUYỂN VỊ cho dầm / khung / dàn theo đúng phong cách trang cũ,
kèm so sánh TRƯỚC tối ưu (nét đứt, màu trắng ở theme tối) và SAU tối ưu (nét liền).

Quy ước dấu (khác trang cũ ở dầm đơn — ghi rõ trong nhãn biểu đồ):
  • Tải xuống > 0 (mũi tên từ trên chĩa xuống).        • M > 0: căng thớ dưới (vẽ về phía dưới, như BMD cũ).
  • V > 0: lực cắt chuẩn (phần bên trái đẩy lên).       • Độ võng hướng xuống > 0.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import legacy_plots as lp
from fem_core import FrameElementResult
from opt_core import StructOptResult, build_beam_model

BC_KINDS = {"ss": ["pin", "roller"], "cantilever": ["fixed", "free"],
            "fixed_pinned": ["fixed", "roller"], "fixed_fixed": ["fixed", "fixed"]}


# ──────────────────────────────────────────────────────
#  THEME
# ──────────────────────────────────────────────────────
def is_dark() -> bool:
    try:
        import streamlit as st
        t = st.context.theme.type
        return t != "light"
    except Exception:
        return True


def before_color() -> str:
    """Màu nét 'trước tối ưu': trắng trên nền tối, gần đen trên nền sáng."""
    return "#ffffff" if is_dark() else "#1f2328"


def sync_theme() -> None:
    lp.BASE_FRAME_COLOR = "#eef3fb" if is_dark() else "#aab2bd"


def _legend(fig, y=1.17):
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=y, x=0.5, xanchor="center", font=dict(size=11)))
    return fig


# ──────────────────────────────────────────────────────
#  DẦM
# ──────────────────────────────────────────────────────
def beam_fields(r) -> dict:
    """V, M dọc dầm và độ võng (hướng xuống > 0) cho thiết kế tối ưu và thiết kế đều (cùng ràng buộc)."""
    model, xs, ctrl = build_beam_model(r.prob)
    out = {"xs": xs, "L": r.prob.L, "model": model}
    for key, A in (("opt", np.asarray(r.A_opt, float)), ("uni", np.full(model.n_el, float(r.A_uniform)))):
        U = model.solve(A)
        d = model.element_diagram(U, A, npts=5)
        out[f"x_{key}"] = np.concatenate([e["x"] for e in d])
        out[f"V_{key}"] = np.concatenate([e["V"] for e in d])
        out[f"M_{key}"] = np.concatenate([e["M"] for e in d])
        out[f"w_{key}"] = -U[1::3]
        out[f"R_{key}"] = model.reactions(U) if key == "opt" else None
    return out


def beam_reactions(r, fields) -> list[tuple[str, str]]:
    xs = fields["xs"]
    items = []
    for dof, v in sorted(fields["R_opt"].items()):
        node, comp = dof // 3, dof % 3
        if comp == 0:
            continue
        name = "R" if comp == 1 else "M"
        unit = "kN" if comp == 1 else "kNm"
        items.append((f"{name} tại x = {xs[node]:.2f} m", f"{v:+.2f} {unit}"))
    return items


def plot_beam_loads(prob):
    L = prob.L
    fig = lp._cb_draw_base_beam_and_supports(L, [L], BC_KINDS[prob.bc])
    fig.update_layout(title=dict(text="<b>Sơ đồ tải trọng</b>", x=0.5, xanchor="center", font=dict(size=14)))
    for q, x1, x2 in prob.udls:
        if abs(q) < 1e-12:
            continue
        down = q > 0
        y_load = 0.58 if down else -0.58
        n = int(np.clip(math.ceil((x2 - x1) / max(L / 18, 1e-9)) + 1, 3, 26))
        fig.add_trace(go.Scatter(x=[x1, x2, x2, x1, x1], y=[0, 0, y_load, y_load, 0], fill="toself", mode="lines",
                                 line={"color": "#168f2c", "width": 1.2}, fillcolor="rgba(22,143,44,0.16)",
                                 hovertemplate=f"UDL: {q:g} kN/m<extra></extra>", showlegend=False))
        for xv in np.linspace(x1, x2, n):
            fig.add_annotation(x=xv, y=0.02 if down else -0.02, ax=xv, ay=y_load, xref="x", yref="y", axref="x",
                               ayref="y", showarrow=True, arrowhead=2, arrowsize=0.95, arrowwidth=1.5,
                               arrowcolor="#168f2c", text="")
        fig.add_annotation(x=(x1 + x2) / 2, y=0.72 if down else -0.72, text=f"{q:g} kN/m", showarrow=False,
                           font={"size": 11, "color": "#168f2c"})
    for P, xp in prob.point_loads:
        down = P > 0
        fig.add_annotation(x=xp, y=0.05 if down else -0.05, ax=xp, ay=0.78 if down else -0.78, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=2.2,
                           arrowcolor=lp.COLOR_SFD, text="")
        fig.add_annotation(x=xp, y=0.92 if down else -0.92, text=f"{P:g} kN", showarrow=False,
                           font={"size": 11, "color": lp.COLOR_SFD})
    return fig


def plot_beam_sfd(f):
    bc = before_color()
    rng = lp._padded_range(np.concatenate([f["V_opt"], f["V_uni"]]))
    fig = lp.synced_figure("Biểu đồ lực cắt V (kN)", f["L"], y_range=rng, y_title="V (kN)", reverse_y=False)
    fig.add_trace(go.Scatter(x=f["x_uni"], y=f["V_uni"], mode="lines", name="Trước tối ưu (dầm đều)",
                             line={"color": bc, "width": 2, "dash": "dash"},
                             hovertemplate="x=%{x:.2f}m  V=%{y:.2f}kN<extra>trước</extra>"))
    fig.add_trace(go.Scatter(x=f["x_opt"], y=f["V_opt"], mode="lines", name="Sau tối ưu", fill="tozeroy",
                             line={"color": lp.COLOR_SFD, "width": 2}, fillcolor="rgba(11,95,255,0.20)",
                             hovertemplate="x=%{x:.2f}m  V=%{y:.2f}kN<extra>sau</extra>"))
    fig.update_yaxes(fixedrange=True)
    return _legend(fig)


def plot_beam_bmd(f):
    bc = before_color()
    rng = lp._padded_range(np.concatenate([f["M_opt"], f["M_uni"]]))
    fig = lp.synced_figure("Biểu đồ mômen uốn M (kNm)", f["L"], y_range=rng, y_title="M (kNm)", reverse_y=True)
    fig.add_trace(go.Scatter(x=f["x_uni"], y=f["M_uni"], mode="lines", name="Trước tối ưu (dầm đều)",
                             line={"color": bc, "width": 2, "dash": "dash"},
                             hovertemplate="x=%{x:.2f}m  M=%{y:.2f}kNm<extra>trước</extra>"))
    fig.add_trace(go.Scatter(x=f["x_opt"], y=f["M_opt"], mode="lines", name="Sau tối ưu", fill="tozeroy",
                             line={"color": lp.COLOR_BMD, "width": 2}, fillcolor="rgba(255,43,43,0.22)",
                             hovertemplate="x=%{x:.2f}m  M=%{y:.2f}kNm<extra>sau</extra>"))
    fig.update_yaxes(autorange="reversed", fixedrange=True)
    return _legend(fig)


def plot_beam_elastic(prob, f):
    bc = before_color()
    L = prob.L
    fig = lp._cb_draw_base_beam_and_supports(L, [L], BC_KINDS[prob.bc])
    for tr in fig.data:
        tr.showlegend = False
    fig.update_layout(title=dict(text="<b>Đường đàn hồi (độ võng phóng đại)</b>", x=0.5, xanchor="center",
                                 font=dict(size=14)))
    mw = max(float(np.max(np.abs(f["w_opt"]))), float(np.max(np.abs(f["w_uni"]))), 1e-30)
    k = 0.65 / mw
    fig.add_trace(go.Scatter(x=f["xs"], y=-f["w_uni"] * k, mode="lines", name="Trước tối ưu (dầm đều)",
                             line={"color": bc, "width": 2.5, "dash": "dash"}, customdata=f["w_uni"] * 1e3,
                             hovertemplate="x=%{x:.2f}m  w=%{customdata:.2f}mm<extra>trước</extra>"))
    fig.add_trace(go.Scatter(x=f["xs"], y=-f["w_opt"] * k, mode="lines", name="Sau tối ưu",
                             line={"color": lp.COLOR_ELAST, "width": 4}, customdata=f["w_opt"] * 1e3,
                             hovertemplate="x=%{x:.2f}m  w=%{customdata:.2f}mm<extra>sau</extra>"))
    i = int(np.argmax(np.abs(f["w_opt"])))
    fig.add_annotation(x=f["xs"][i], y=-f["w_opt"][i] * k, text=f"{abs(f['w_opt'][i]) * 1e3:.2f} mm", showarrow=True,
                       ay=34, arrowcolor=lp.COLOR_ELAST, font=dict(size=11, color=lp.COLOR_ELAST))
    return _legend(fig)


# ──────────────────────────────────────────────────────
#  KHUNG / DÀN
# ──────────────────────────────────────────────────────
def struct_frames(res: StructOptResult):
    inp = res.inp
    df_nodes = pd.DataFrame(list(inp.nodes), columns=["x (m)", "y (m)"])
    q = list(inp.member_q) + [0.0] * (len(inp.members) - len(inp.member_q))
    df_el = pd.DataFrame({"i": [i for i, _ in inp.members], "j": [j for _, j in inp.members],
                          "udl_local": q if inp.mode == "frame" else [0.0] * len(inp.members)})
    df_sup = pd.DataFrame([dict(node=n, ux=bool(a), uy=bool(b), rz=bool(c)) for n, a, b, c in inp.supports])
    loads = inp.user_nodal if (inp.mode == "frame" and inp.user_nodal) else inp.loads
    df_nload = pd.DataFrame([dict(node=n, **{"Fx (kN)": fx, "Fy (kN)": fy, "Mz (kNm)": mz})
                             for n, fx, fy, mz in loads]) if loads else None
    return df_nodes, df_el, df_sup, df_nload


def struct_element_results(res: StructOptResult):
    """Gộp các phần tử con của mỗi thanh gốc thành 1 FrameElementResult (nhãn trị số chỉ ở đầu thanh + cực trị)."""
    model, opt, inp = res.model, res.opt, res.inp
    out = []
    if inp.mode == "truss":
        for k, (i, j) in enumerate(inp.members):
            xg = np.array([inp.nodes[i][0], inp.nodes[j][0]]); yg = np.array([inp.nodes[i][1], inp.nodes[j][1]])
            N = float(opt.N[res.member_of_el == k][0])
            out.append(FrameElementResult(k, np.array([0.0, math.hypot(xg[1] - xg[0], yg[1] - yg[0])]),
                                          np.array([N, N]), np.zeros(2), np.zeros(2), xg, yg))
        return out
    diag = model.element_diagram(opt.U, opt.A, npts=11)
    for k in range(len(inp.members)):
        sel = np.where(res.member_of_el == k)[0]
        X, Y, N, V, M = [], [], [], [], []
        for n_, e in enumerate(sel):
            d = diag[e]
            sl = slice(0, None) if n_ == 0 else slice(1, None)
            X += list(d["x"][sl]); Y += list(d["y"][sl]); N += list(d["N"][sl]); V += list(d["V"][sl]); M += list(d["M"][sl])
        X, Y = np.array(X), np.array(Y)
        s_loc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
        out.append(FrameElementResult(k, s_loc, np.array(N), np.array(V), np.array(M), X, Y))
    return out


def plot_struct_loads(res: StructOptResult):
    sync_theme()
    df_nodes, df_el, df_sup, df_nload = struct_frames(res)
    fig = lp._pf_geometry_plot(df_nodes, df_el, df_sup, None, df_nload)
    fig.update_layout(title="Sơ đồ kết cấu và tải trọng", height=500,
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def plot_struct_diagram(res: StructOptResult, kind: str):
    sync_theme()
    df_nodes, df_el, df_sup, _ = struct_frames(res)
    title = {"moment": "Biểu đồ mômen M (kNm)", "shear": "Biểu đồ lực cắt V (kN)",
             "axial": "Biểu đồ lực dọc N (kN) — dương: kéo"}[kind]
    rr = SimpleNamespace(element_results=struct_element_results(res))
    return lp._pf_frame_diagram(df_nodes, df_el, df_sup, rr, kind, title)


def deformed_shape(res: StructOptResult, npts: int = 17):
    """Điểm hình học đã biến dạng (chưa phóng đại) cho từng phần tử: list[(x0,y0,dx,dy)]."""
    model, U = res.model, res.opt.U
    out = []
    for e in range(model.n_el):
        i, j = model.conn[e]
        p0, p1 = model.xy[i], model.xy[j]
        xi = np.linspace(0, 1, npts)
        x0 = p0[0] + xi * (p1[0] - p0[0]); y0 = p0[1] + xi * (p1[1] - p0[1])
        if model.kind == "frame":
            c, s, L = model.c[e], model.s[e], model.L[e]
            ul = model.T[e] @ U[model.dofs[e]]
            u = ul[0] * (1 - xi) + ul[3] * xi
            v = ((1 - 3 * xi ** 2 + 2 * xi ** 3) * ul[1] + L * xi * (1 - xi) ** 2 * ul[2]
                 + (3 * xi ** 2 - 2 * xi ** 3) * ul[4] + L * xi ** 2 * (xi - 1) * ul[5])
            dx = c * u - s * v; dy = s * u + c * v
        else:
            ue = U[model.dofs[e]]
            dx = ue[0] * (1 - xi) + ue[2] * xi; dy = ue[1] * (1 - xi) + ue[3] * xi
        out.append((x0, y0, dx, dy))
    return out


def plot_struct_deformed(res: StructOptResult):
    sync_theme()
    inp = res.inp
    bc = before_color()
    df_nodes, df_el, df_sup, _ = struct_frames(res)
    shapes = deformed_shape(res)
    xs = [p[0] for p in inp.nodes]; ys = [p[1] for p in inp.nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    dmax = max(float(np.max(np.hypot(dx, dy))) for _, _, dx, dy in shapes)
    mag = 0.07 * span / dmax if dmax > 0 else 1.0
    fig = go.Figure()
    # bản gốc (trước biến dạng) — nét đứt
    ux, uy = [], []
    for i, j in inp.members:
        ux += [inp.nodes[i][0], inp.nodes[j][0], None]; uy += [inp.nodes[i][1], inp.nodes[j][1], None]
    fig.add_trace(go.Scatter(x=ux, y=uy, mode="lines", name="Chưa biến dạng", line=dict(color=bc, width=2, dash="dash"),
                             hoverinfo="skip"))
    # sau biến dạng
    for e, (x0, y0, dx, dy) in enumerate(shapes):
        fig.add_trace(go.Scatter(x=x0 + mag * dx, y=y0 + mag * dy, mode="lines", name="Sau biến dạng",
                                 showlegend=(e == 0), line=dict(color=lp.COLOR_ELAST, width=4),
                                 customdata=np.hypot(dx, dy) * 1e3,
                                 hovertemplate="δ = %{customdata:.2f} mm<extra></extra>"))
    # nhãn nút có chuyển vị lớn nhất
    nd = res.model.node_disp(res.opt.U)
    mags = np.hypot(nd[:, 0], nd[:, 1])
    k = int(np.argmax(mags))
    px, py = res.model.xy[k]
    fig.add_annotation(x=px + mag * nd[k, 0], y=py + mag * nd[k, 1], ax=40, ay=-40, showarrow=True,
                       arrowcolor=lp.COLOR_ELAST, font=dict(size=12, color=lp.COLOR_ELAST),
                       text=f"δmax = {mags[k] * 1e3:.2f} mm (nút {k})")
    sup_size = max(span * 0.035, 0.25)
    for _, row in df_sup.iterrows():
        n = int(row["node"])
        lp.draw_support_pf(fig, float(inp.nodes[n][0]), float(inp.nodes[n][1]), lp._pf_support_type(row), sup_size)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    pad = 0.12 * span
    fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad]); fig.update_yaxes(range=[min(ys) - pad, max(ys) + 2 * pad])
    fig.update_layout(title=f"Chuyển vị — hình dạng biến dạng phóng đại ×{mag:,.0f}", height=500,
                      template="plotly_white", margin=dict(l=50, r=20, t=90, b=40),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0))
    return fig
