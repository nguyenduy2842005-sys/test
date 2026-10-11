"""
Đớp Tool — Tối ưu hoá kết cấu dàn/khung thép bằng Phép tính Biến phân
Chạy:  streamlit run web_app.py
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from fem_core import build_docx_bytes
from opt_core import (
    BEAM_BCS, COLUMN_BCS, STEEL_E, STEEL_RHO, STEEL_SIGMA,
    BeamProblem, StructInput, make_family, optimize_beam, optimize_column, optimize_frame,
)

st.set_page_config(page_title="Đớp Tool", page_icon="🦈", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#e67300", "#2ca02c", "#d62728", "#7f7f7f"


# ══════════════════════════════════════════════════════
#  CSS & tiện ích
# ══════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    #MainMenu, footer {display:none !important;}
    .block-container {padding-top: 1.2rem; max-width: 1600px;}
    .metric-strip {display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:.3rem 0 .9rem;}
    .metric-card {border:1px solid rgba(128,128,128,.28); border-radius:8px; padding:10px 12px;}
    .metric-label {opacity:.7; font-size:.78rem; margin-bottom:3px;}
    .metric-value {font-weight:700; font-size:1.12rem;}
    .metric-good .metric-value {color:#2ca02c;}
    .metric-bad .metric-value {color:#d62728;}
    textarea {font-family: Consolas,"Courier New",monospace !important;}
    </style>""", unsafe_allow_html=True)


def metrics(items):
    html = '<div class="metric-strip">'
    for it in items:
        label, value = it[0], it[1]
        cls = it[2] if len(it) > 2 else ""
        html += f'<div class="metric-card {cls}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def clean_rows(df, ncols, int_cols=()):
    out = []
    if df is None:
        return out
    for _, row in df.iterrows():
        vals = list(row.values[:ncols])
        if any(pd.isna(v) for v in vals):
            continue
        out.append(tuple(int(v) if k in int_cols else float(v) for k, v in enumerate(vals)))
    return out


def report_block(text: str, key: str, title: str):
    with st.expander("📄 Thuyết minh tính toán", expanded=False):
        st.text_area("Nội dung", text, height=420, key=f"ta_{key}", label_visibility="collapsed")
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Tải .txt", text.encode("utf-8"), file_name=f"{key}.txt", key=f"dl_txt_{key}")
        try:
            c2.download_button("⬇️ Tải .docx", build_docx_bytes(text, title), file_name=f"{key}.docx",
                               key=f"dl_docx_{key}")
        except Exception as ex:  # python-docx thiếu
            c2.caption(f"Không tạo được .docx: {ex}")


def family_selector(prefix: str, default="Ống tròn mỏng"):
    opts = ["Chữ nhật đặc", "Tròn đặc", "Ống tròn mỏng", "Chữ I (cánh mỏng, h cố định)", "Tuỳ chỉnh I = α·Aⁿ"]
    idx = opts.index(default)
    c1, c2 = st.columns([2, 1.2])
    kind = c1.selectbox("Họ tiết diện", opts, index=idx, key=f"{prefix}_fam",
                        help="Biến thiết kế duy nhất là diện tích A(x); I(x) và W(x) suy ra từ họ tiết diện (I = α·Aⁿ).")
    if kind == "Chữ nhật đặc":
        r = c2.number_input("b/h", 0.1, 2.0, 0.5, 0.05, key=f"{prefix}_r")
        return make_family("rect", ratio=r)
    if kind == "Tròn đặc":
        return make_family("circle")
    if kind == "Ống tròn mỏng":
        r = c2.number_input("t/D", 0.01, 0.2, 0.05, 0.01, key=f"{prefix}_t")
        return make_family("tube", ratio=r)
    if kind.startswith("Chữ I"):
        h = c2.number_input("h (m)", 0.05, 3.0, 0.4, 0.05, key=f"{prefix}_h")
        return make_family("ibeam", h=h)
    cc = st.columns(4)
    n = cc[0].number_input("n", 0.5, 3.0, 2.0, 0.1, key=f"{prefix}_n")
    al = cc[1].number_input("α", 1e-4, 10.0, 0.08, 0.01, format="%.4f", key=f"{prefix}_al")
    ga = cc[2].number_input("γ", 0.0, 1.5, 0.5, 0.05, key=f"{prefix}_ga")
    be = cc[3].number_input("β", 1e-4, 10.0, 0.17, 0.01, format="%.4f", key=f"{prefix}_be")
    return make_family("custom", n=n, alpha=al, gamma=ga, beta=be)


def fig_layout(fig, title="", h=380, xt="", yt=""):
    fig.update_layout(title=title, height=h, margin=dict(l=50, r=20, t=50, b=40), template="plotly_white",
                      legend=dict(orientation="h", y=1.12, x=0), xaxis_title=xt, yaxis_title=yt)
    return fig


# ══════════════════════════════════════════════════════
#  TAB 0 — LÝ THUYẾT
# ══════════════════════════════════════════════════════
def tab_theory():
    st.header("Cơ sở toán học")
    st.markdown("""
**Ý tưởng.** Khi biến thiết kế là *hàm liên tục* $A(x)$ (dầm vút thon, cột biến tiết diện) ta không tìm một vài con số mà tìm *cả một hàm*.
Đó là bài toán cực trị phiếm hàm. Đớp Tool dùng phép tính biến phân để ra **dạng đóng** của $A^*(x)$, rồi dùng FEM + tiêu chuẩn tối ưu để
(i) kiểm chứng và (ii) mở rộng cho các sơ đồ siêu tĩnh, khung và dàn, nơi không còn nghiệm đóng.
""")
    st.subheader("1. Phương trình Euler–Lagrange")
    st.latex(r"J[y]=\int_{0}^{L}F(x,y,y')\,\mathrm dx\ \to\ \text{cực trị}\quad\Rightarrow\quad F_y-\frac{\mathrm d}{\mathrm dx}\big(F_{y'}\big)=0")
    st.markdown(r"""
Với ràng buộc tích phân (thể tích, độ võng) thêm nhân tử Lagrange $\lambda$: $\ \tilde F=F+\lambda G$.
**Trường hợp suy biến quan trọng:** khi $F$ *không chứa* $y'$ thì $F_{y'}=0$ và E–L rút về **phương trình đại số** $F_y=0$ — nghiệm tìm được ngay, không cần giải ODE.
""")
    st.subheader("2. Dầm vút thon — cực tiểu thể tích với ràng buộc độ võng")
    st.markdown(r"Họ tiết diện: $I=\alpha A^{n}$ (chữ nhật/tròn/ống đồng dạng $n=2$; chữ I cánh mỏng với $h$ cố định $n=1$). "
                r"Độ võng tại điểm khống chế $x_p$ theo nguyên lý công ảo:")
    st.latex(r"\delta(x_p)=\int_0^L\frac{M(x)\,\bar m(x)}{E\,\alpha\,A^{n}(x)}\,\mathrm dx=\delta_{\max},\qquad V=\int_0^L A\,\mathrm dx\to\min")
    st.latex(r"\tilde F=A+\lambda\,\frac{M\bar m}{E\alpha A^{n}}\ \ (\text{không chứa }A')\ \Rightarrow\ 1-\frac{\lambda n M\bar m}{E\alpha A^{n+1}}=0")
    st.latex(r"\boxed{A^{*}(x)=\Big[\tfrac{\lambda n}{E\alpha}\,M(x)\,\bar m(x)\Big]^{\frac1{n+1}}}")
    st.markdown(r"""
- Tải tập trung tại điểm khống chế: $\bar m\propto M\Rightarrow A^*\propto|M|^{\frac{2}{n+1}}$ (chữ nhật/tròn: $|M|^{2/3}$; chữ I cánh mỏng: $|M|$).
- Kết hợp điều kiện bền $A\ge A_{bền}(x)=A_W(|M|/[\sigma])$ và $A\ge A_{\min}$: nghiệm KKT là $A=\max(A_{bền},A_{\min},c\,g^{1/(n+1)})$, $g=\max(M\bar m,0)$; hằng số $c$ giải từ $\delta(c)=\delta_{\max}$.
- Dầm siêu tĩnh: $M$ phụ thuộc $A$ nên không có dạng đóng → dùng FEM + OC (mục 4).
""")
    st.subheader("3. Cột — cực đại tải tới hạn với thể tích cho trước (Lagrange – Keller – Tadjbakhsh)")
    st.latex(r"P_{cr}=\min_w\frac{\int E\alpha A^{n}w''^2\,\mathrm dx}{\int w'^2\,\mathrm dx},\quad \int A\,\mathrm dx=V")
    st.latex(r"\delta_A:\ \ n\,E\alpha A^{n-1}w''^{2}=\lambda\ \ \ (\text{đại số}),\qquad M=-Pw\ \Rightarrow\ A\propto|w|^{\frac2{n+1}}")
    st.latex(r"w''=-\kappa\,|w|^{\frac{1-n}{n+1}}\,\mathrm{sgn}\,w\ \xrightarrow{\ u=A/A_{\max}\ }\ \xi=I_u\!\Big(\tfrac{n+1}2,\tfrac12\Big)\ \Rightarrow\ u(\xi)=I^{-1}_{\xi}\!\Big(\tfrac{n+1}2,\tfrac12\Big)")
    st.latex(r"\frac{\bar A}{A_{\max}}=\frac{n+1}{n+2},\qquad P_{cr}^*=\frac{(n+1)\,B^{2}\!\big(\tfrac{n+1}2,\tfrac12\big)\,E\alpha\,A_{\max}^{\,n}}{L_{ef}^{2}}")
    st.markdown(r"Với $n=2$, hai đầu khớp: $P^*_{cr}/P_{cr,đều}=4/3$ — **kết quả cổ điển của Keller (1960)**, được dùng làm test kiểm chứng. "
                r"Nghiệm E–L cho $A\to0$ tại điểm mômen bằng 0; thực tế phải áp cận dưới $A_{\min}$.")
    st.subheader("4. Phần số: tiêu chuẩn tối ưu (OC) và KKT trên lưới FEM")
    st.latex(r"\min_{A}\ W=\rho\sum A_eL_e\quad\text{s.t.}\quad \frac{|N_e|}{A_e\sigma_N}+\frac{|M_e|}{W_e[\sigma]}\le1,\ \ |\delta_k|\le\delta_k^{\lim}")
    st.latex(r"\frac{\partial\mathcal L}{\partial A_e}=0\ \Leftrightarrow\ \rho L_e=\sum_k\lambda_k\frac{\partial g_k}{\partial A_e}\ \ \ (\text{dạng rời rạc của Euler–Lagrange}),\qquad \frac{\partial\delta}{\partial A_e}=-\,\mathbf u_{unit}^{T}\frac{\partial\mathbf K}{\partial A_e}\mathbf u")
    st.markdown("""
Quy trình: (1) FEM nhanh dùng ma trận phần tử / tải tương đương của `fem_core.py` (đối chiếu với `solve_plane_frame` → sai khác ~1e-12);
(2) OC/FSD tạo nghiệm khởi tạo; (3) vòng ngoài giữ cận bền (FSD), vòng trong SLSQP với đạo hàm liên hợp (adjoint) cho ràng buộc chuyển vị;
(4) báo cáo sai số KKT rời rạc. Dàn 10 thanh chuẩn (Schmit) cho 5076,7 lb, khớp mức tham chiếu ~5060 lb.
""")
    st.info("**Giới hạn:** Euler–Bernoulli, đàn hồi tuyến tính, bỏ qua biến dạng cắt, ổn định cục bộ, P-Δ, mỏi, tải bản thân và chi tiết liên kết. "
            "Tiết diện thực phải làm tròn theo thép hình rồi kiểm tra lại theo TCVN/AISC. Nghiệm với kết cấu siêu tĩnh là cực trị địa phương.")


# ══════════════════════════════════════════════════════
#  TAB 1 — CỘT
# ══════════════════════════════════════════════════════
def tab_column():
    st.header("① Cấu kiện — Cột chịu nén biến tiết diện")
    left, right = st.columns([1, 2.2])
    with left:
        bc = st.selectbox("Sơ đồ liên kết", list(COLUMN_BCS), format_func=COLUMN_BCS.get, key="c_bc")
        L = st.number_input("Chiều dài L (m)", 0.5, 100.0, 5.0, 0.5, key="c_L")
        E = st.number_input("E (GPa)", 1.0, 400.0, STEEL_E / 1e6, 5.0, key="c_E") * 1e6
        fam = family_selector("c")
        mode = st.radio("Bài toán", ["Cho thể tích → cực đại P_cr", "Cho P yêu cầu → cực tiểu thể tích"], key="c_mode")
        if mode.startswith("Cho thể"):
            A0 = st.number_input("Diện tích trung bình A₀ = V/L (cm²)", 1.0, 5000.0, 30.0, 1.0, key="c_A0")
            P_req = None
        else:
            P_req = st.number_input("P yêu cầu (kN)", 1.0, 1e7, 500.0, 10.0, key="c_P")
        floor = st.slider("Cận dưới A_min (% của A_max)", 0.0, 20.0, 1.0, 0.5, key="c_floor") / 100.0
        run_num = st.checkbox("Chạy thêm tối ưu số OC (kiểm chứng nghiệm đóng)", True, key="c_num")
        go_btn = st.button("🚀 Tối ưu cột", type="primary", key="c_go")
    if go_btn:
        try:
            with st.spinner("Đang tính..."):
                if P_req is None:
                    st.session_state["c_res"] = optimize_column(L, fam, E, bc, V=A0 * 1e-4 * L, floor_frac=floor,
                                                                run_numeric=run_num)
                else:
                    st.session_state["c_res"] = optimize_column(L, fam, E, bc, P_req=P_req, floor_frac=floor,
                                                                run_numeric=run_num)
        except Exception as ex:
            st.error(f"Lỗi: {ex}")
    r = st.session_state.get("c_res")
    with right:
        if r is None:
            st.info("Nhập thông số rồi bấm **Tối ưu cột**.")
            return
        metrics([
            ("P_cr tối ưu (FEM)", f"{r.P_fem_opt:,.1f} kN"),
            ("P_cr cột đều", f"{r.P_uniform:,.1f} kN"),
            ("Hệ số tăng tải (FEM)", f"×{r.ratio_fem:.3f}", "metric-good"),
            ("Hệ số tăng tải (đóng, không cận)", f"×{r.ratio_closed:.3f}"),
            ("V = ∫A dx", f"{r.V * 1e6:,.0f} cm²·m→ {r.V:.5f} m³"),
            ("A_max", f"{np.max(r.A_opt) * 1e4:.2f} cm²"),
        ])
        fig = go.Figure()
        fig.add_scatter(x=r.x / r.L, y=r.A_opt * 1e4, name="E–L nghiệm đóng (FEM kiểm)", line=dict(color=BLUE, width=3))
        if r.A_numeric is not None:
            fig.add_scatter(x=r.x_numeric / r.L, y=r.A_numeric * 1e4, name="OC số", mode="lines",
                            line=dict(color=ORANGE, dash="dash"))
        fig.add_scatter(x=[0, 1], y=[r.A_uniform * 1e4] * 2, name="Cột đều", line=dict(color=GREY, dash="dot"))
        st.plotly_chart(fig_layout(fig, "Diện tích tối ưu A(x)", 360, "x/L", "A (cm²)"), width="stretch")
        c1, c2 = st.columns(2)
        sil = go.Figure()
        rad = np.sqrt(np.maximum(r.A_opt, 0.0)) / np.sqrt(np.max(r.A_opt))
        sil.add_scatter(x=np.concatenate([r.x, r.x[::-1]]), y=np.concatenate([rad, -rad[::-1]]), fill="toself",
                        fillcolor="rgba(31,119,180,.35)", line=dict(color=BLUE), name="Cột tối ưu")
        sil.update_yaxes(scaleanchor=None, showticklabels=False)
        c1.plotly_chart(fig_layout(sil, "Hình dạng (bề rộng ∝ √A)", 320, "x (m)", ""), width="stretch")
        if r.history:
            cv = go.Figure()
            cv.add_scatter(y=np.array(r.history) / r.P_closed, name="P_cr(OC)/P_closed", line=dict(color=ORANGE))
            cv.add_hline(y=1.0, line_dash="dot")
            c2.plotly_chart(fig_layout(cv, "Hội tụ OC so với nghiệm đóng", 320, "vòng lặp", "P/P*"), width="stretch")
        report_block(r.report, "cot_toi_uu", "Thuyết minh — Tối ưu cột")


# ══════════════════════════════════════════════════════
#  TAB 2 — DẦM
# ══════════════════════════════════════════════════════
def tab_beam():
    st.header("① Cấu kiện — Dầm vút thon A(x)")
    left, right = st.columns([1.1, 2.2])
    with left:
        bc = st.selectbox("Sơ đồ", list(BEAM_BCS), format_func=BEAM_BCS.get, key="b_bc")
        L = st.number_input("Chiều dài L (m)", 1.0, 100.0, 8.0, 0.5, key="b_L")
        c1, c2 = st.columns(2)
        E = c1.number_input("E (GPa)", 1.0, 400.0, STEEL_E / 1e6, 5.0, key="b_E") * 1e6
        sig = c2.number_input("[σ] (MPa)", 10.0, 1000.0, STEEL_SIGMA / 1e3, 10.0, key="b_sig") * 1e3
        fam = family_selector("b", default="Chữ I (cánh mỏng, h cố định)")
        st.caption("Tải trọng (hướng xuống > 0)")
        pl_df = st.data_editor(pd.DataFrame({"P (kN)": [100.0], "x (m)": [L / 2]}), num_rows="dynamic",
                               key="b_pl", hide_index=True, width="stretch")
        udl_df = st.data_editor(pd.DataFrame({"q (kN/m)": [10.0], "x1 (m)": [0.0], "x2 (m)": [L]}), num_rows="dynamic",
                                key="b_udl", hide_index=True, width="stretch")
        use_d = st.checkbox("Ràng buộc độ võng", True, key="b_usd")
        d1, d2 = st.columns(2)
        dmax = d1.number_input("|δ| max (mm)", 0.1, 1000.0, round(L * 1000 / 400, 1), 1.0, key="b_dmax") / 1e3
        xc = d2.number_input("Vị trí khống chế x (m)", 0.0, L, L / 2 if bc != "cantilever" else L, 0.5, key="b_xc")
        amin = st.number_input("A_min (cm²)", 0.01, 1000.0, 5.0, 0.5, key="b_amin") * 1e-4
        nel = st.slider("Số phần tử lưới", 40, 200, 100, 10, key="b_nel")
        go_btn = st.button("🚀 Tối ưu dầm", type="primary", key="b_go")
    if go_btn:
        try:
            pls = clean_rows(pl_df, 2)
            udls = clean_rows(udl_df, 3)
            for P, x in pls:
                if not 0 <= x <= L:
                    raise ValueError(f"Tải tập trung tại x={x} nằm ngoài [0, {L}].")
            for q, a, b in udls:
                if not (0 <= a < b <= L):
                    raise ValueError(f"UDL cần 0 ≤ x1 < x2 ≤ {L}.")
            if not pls and not udls:
                raise ValueError("Chưa có tải trọng.")
            prob = BeamProblem(L=L, bc=bc, udls=udls, point_loads=pls, fam=fam, E=E, sigma_a=sig,
                               delta_max=dmax if use_d else None, x_ctrl=xc, A_min=amin, n_elem=nel)
            with st.spinner("Đang tối ưu (FEM + KKT)..."):
                st.session_state["b_res"] = optimize_beam(prob)
        except Exception as ex:
            st.error(f"Lỗi: {ex}")
    r = st.session_state.get("b_res")
    with right:
        if r is None:
            st.info("Nhập thông số rồi bấm **Tối ưu dầm**.")
            return
        sav = (1 - r.vol_opt / r.vol_uniform) * 100
        items = [("Thể tích tối ưu", f"{r.vol_opt:.4f} m³"), ("Thể tích dầm đều", f"{r.vol_uniform:.4f} m³"),
                 ("Tiết kiệm vật liệu", f"{sav:.1f}%", "metric-good")]
        if r.vol_closed is not None:
            items.append(("Nghiệm đóng E–L", f"{r.vol_closed:.4f} m³"))
        if r.delta_ctrl_opt is not None:
            items.append(("δ tại điểm khống chế", f"{r.delta_ctrl_opt * 1e3:.2f} / {r.prob.delta_max * 1e3:.2f} mm"))
        items.append(("Ràng buộc", "Thoả" if r.opt.feasible else "KHÔNG thoả", "metric-good" if r.opt.feasible else "metric-bad"))
        metrics(items)
        fig = go.Figure()
        xs_step = np.repeat(r.xs, 2)[1:-1]
        fig.add_scatter(x=xs_step, y=np.repeat(r.A_opt, 2) * 1e4, name="FEM + KKT", line=dict(color=BLUE, width=3))
        if r.closed is not None:
            fig.add_scatter(x=r.closed["x"], y=r.closed["A"] * 1e4, name="Nghiệm đóng E–L", line=dict(color=ORANGE, dash="dash"))
        fig.add_scatter(x=[0, r.prob.L], y=[r.A_uniform * 1e4] * 2, name="Dầm đều", line=dict(color=GREY, dash="dot"))
        st.plotly_chart(fig_layout(fig, "Diện tích tối ưu A(x)", 360, "x (m)", "A (cm²)"), width="stretch")
        c1, c2 = st.columns(2)
        fm = go.Figure()
        fm.add_scatter(x=r.moment_x, y=r.moment_opt, fill="tozeroy", line=dict(color=RED), name="M")
        fm.update_yaxes(autorange="reversed")
        c1.plotly_chart(fig_layout(fm, "Biểu đồ mômen M (kNm)", 320, "x (m)", "M"), width="stretch")
        fd = go.Figure()
        fd.add_scatter(x=r.xs, y=r.deflection_opt * 1e3, name="Tối ưu", line=dict(color=BLUE))
        fd.add_scatter(x=r.xs, y=r.deflection_uni * 1e3, name="Dầm đều", line=dict(color=GREY, dash="dot"))
        fd.update_yaxes(autorange="reversed")
        c2.plotly_chart(fig_layout(fd, "Độ võng (mm)", 320, "x (m)", "w"), width="stretch")
        report_block(r.report, "dam_toi_uu", "Thuyết minh — Tối ưu dầm vút thon")


# ══════════════════════════════════════════════════════
#  TAB 3 — KHUNG / DÀN
# ══════════════════════════════════════════════════════
@st.cache_resource
def _canvas_component():
    return components.declare_component("dop_canvas", path=os.path.join(HERE, "canvas"))


def _sup(node, kind):
    t = {"pin": (True, True, False), "rollx": (False, True, False), "fixed": (True, True, True)}[kind]
    return dict(node=node, ux=t[0], uy=t[1], rz=t[2])


def _truss_warren():
    nodes = [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (1.5, 2.5), (4.5, 2.5), (7.5, 2.5), (10.5, 2.5)]
    mem = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
           (0, 5), (5, 1), (1, 6), (6, 2), (2, 7), (7, 3), (3, 8), (8, 4)]
    return dict(mode="Dàn (nút khớp)", nodes=nodes, members=mem, supports=[_sup(0, "pin"), _sup(4, "rollx")],
                nodal=[(5, 0, -50), (6, 0, -50), (7, 0, -50), (8, 0, -50)], udl=[], limits=[(2, "uy", 30.0)], seg=1)


def _frame_portal():
    nodes = [(0, 0), (0, 4), (3, 5), (6, 4), (6, 0)]
    mem = [(0, 1), (1, 2), (2, 3), (3, 4)]
    return dict(mode="Khung phẳng (nút cứng)", nodes=nodes, members=mem,
                supports=[_sup(0, "fixed"), _sup(4, "fixed")], nodal=[(1, 15, 0)],
                udl=[(1, 20.0), (2, 20.0)], limits=[(2, "uy", 20.0)], seg=4)


def _frame_two_story():
    nodes = [(0, 0), (6, 0), (0, 3.5), (6, 3.5), (0, 7), (6, 7)]
    mem = [(0, 2), (2, 4), (1, 3), (3, 5), (2, 3), (4, 5)]
    return dict(mode="Khung phẳng (nút cứng)", nodes=nodes, members=mem,
                supports=[_sup(0, "fixed"), _sup(1, "fixed")], nodal=[(2, 20, 0), (4, 40, 0)],
                udl=[(4, 30.0), (5, 30.0)], limits=[(4, "ux", 35.0)], seg=4)


PRESETS = {"Dàn Warren 12 m": _truss_warren, "Khung cổng 1 nhịp (mái dốc)": _frame_portal,
           "Khung 2 tầng 1 nhịp": _frame_two_story}


def _load_preset(name):
    p = PRESETS[name]()
    st.session_state["geo"] = dict(nodes=[dict(x=float(x), y=float(y)) for x, y in p["nodes"]],
                                   elements=[dict(i=i, j=j) for i, j in p["members"]], supports=p["supports"])
    st.session_state["s_mode"] = p["mode"]
    st.session_state["s_seg"] = p["seg"]
    st.session_state["nodal0"] = pd.DataFrame(p["nodal"], columns=["Nút", "Fx (kN)", "Fy (kN)"]) if p["nodal"] else \
        pd.DataFrame({"Nút": [0], "Fx (kN)": [0.0], "Fy (kN)": [0.0]})
    st.session_state["udl0"] = pd.DataFrame(p["udl"], columns=["Thanh", "w (kN/m, hướng xuống)"]) if p["udl"] else \
        pd.DataFrame({"Thanh": [0], "w (kN/m, hướng xuống)": [0.0]})
    st.session_state["lim0"] = pd.DataFrame(p["limits"], columns=["Nút", "Phương", "|δ| max (mm)"]) if p["limits"] else \
        pd.DataFrame({"Nút": [0], "Phương": ["uy"], "|δ| max (mm)": [30.0]})
    st.session_state["s_token"] = st.session_state.get("s_token", 0) + 1
    st.session_state.pop("s_res", None)


def _init_struct_state():
    if "geo" not in st.session_state:
        _load_preset("Dàn Warren 12 m")


def build_struct_input(geo, mode_label, fam, E, sig, rho, amin, amax, buck, keff, fs, seg, nodal_df, udl_df, lim_df):
    nodes = [(n["x"], n["y"]) for n in geo["nodes"]]
    members = [(e["i"], e["j"]) for e in geo["elements"]]
    if len(nodes) < 2 or not members:
        raise ValueError("Cần ít nhất 2 nút và 1 thanh — hãy vẽ trên canvas hoặc chọn mẫu.")
    is_frame = mode_label.startswith("Khung")
    sup = []
    for s in geo["supports"]:
        sup.append((int(s["node"]), bool(s["ux"]), bool(s["uy"]), bool(s["rz"]) if is_frame else False))
    if not sup:
        raise ValueError("Chưa có gối tựa (chế độ 🔒 Gối tựa: bấm vào nút để đổi loại gối).")
    loads = {}
    for k, fx, fy in clean_rows(nodal_df, 3, int_cols=(0,)):
        if not 0 <= k < len(nodes):
            raise ValueError(f"Tải nút: nút {k} không tồn tại.")
        a = loads.setdefault(k, [0.0, 0.0, 0.0]); a[0] += fx; a[1] += fy
    member_q = [0.0] * len(members)
    for m, w in clean_rows(udl_df, 2, int_cols=(0,)):
        if not 0 <= m < len(members):
            raise ValueError(f"Tải phân bố: thanh {m} không tồn tại.")
        if w == 0:
            continue
        i, j = members[m]
        dx, dy = nodes[j][0] - nodes[i][0], nodes[j][1] - nodes[i][1]
        Lm = math.hypot(dx, dy); c, s = dx / Lm, dy / Lm
        if is_frame:
            member_q[m] += -w * c                     # thành phần vuông góc trục thanh (trục y cục bộ)
            fax = -w * s * Lm / 2.0                   # thành phần dọc trục → gộp 1/2 vào mỗi đầu thanh
            for nd in (i, j):
                a = loads.setdefault(nd, [0.0, 0.0, 0.0]); a[0] += fax * c; a[1] += fax * s
        else:
            for nd in (i, j):
                a = loads.setdefault(nd, [0.0, 0.0, 0.0]); a[1] += -w * Lm / 2.0
    if not any(abs(v) > 0 for a in loads.values() for v in a) and not any(member_q):
        raise ValueError("Chưa có tải trọng.")
    lims = []
    for k, d, v in [(int(r.iloc[0]), str(r.iloc[1]), float(r.iloc[2])) for _, r in lim_df.iterrows()
                    if not (pd.isna(r.iloc[0]) or pd.isna(r.iloc[1]) or pd.isna(r.iloc[2]))]:
        if not 0 <= k < len(nodes):
            raise ValueError(f"Giới hạn chuyển vị: nút {k} không tồn tại.")
        if d not in ("ux", "uy"):
            raise ValueError("Phương chuyển vị phải là ux hoặc uy.")
        if v > 0:
            lims.append((k, d, v / 1e3))
    return StructInput(nodes=nodes, members=members, supports=sup,
                       loads=[(k, a[0], a[1], a[2]) for k, a in loads.items()], member_q=member_q,
                       mode="frame" if is_frame else "truss", fam=fam, E=E, sigma_a=sig, rho=rho,
                       FS_buckling=fs, k_eff=keff, check_buckling=buck, A_min=amin, A_max=amax,
                       disp_limits=lims, segments=seg)


def plot_structure(inp: StructInput, area_member=None, util_member=None, title="Hình học kết cấu"):
    fig = go.Figure()
    xs = [p[0] for p in inp.nodes]; ys = [p[1] for p in inp.nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    vals = area_member
    vmin, vmax = (float(np.min(vals)), float(np.max(vals))) if vals is not None else (0, 1)
    for k, (i, j) in enumerate(inp.members):
        if vals is None:
            col, w = BLUE, 4
        else:
            t = 0.0 if vmax - vmin < 1e-12 else (vals[k] - vmin) / (vmax - vmin)
            col = pc.sample_colorscale("Turbo", 0.1 + 0.8 * t)[0]
            w = 2 + 12 * vals[k] / vmax
        fig.add_scatter(x=[inp.nodes[i][0], inp.nodes[j][0]], y=[inp.nodes[i][1], inp.nodes[j][1]], mode="lines",
                        line=dict(color=col, width=w), hoverinfo="text", showlegend=False,
                        text=f"Thanh {k}: " + (f"A = {vals[k] * 1e4:.2f} cm²" if vals is not None else ""))
        mx, my = (inp.nodes[i][0] + inp.nodes[j][0]) / 2, (inp.nodes[i][1] + inp.nodes[j][1]) / 2
        fig.add_annotation(x=mx, y=my, text=f"{k}" if vals is None else f"{vals[k] * 1e4:.1f}", showarrow=False,
                           font=dict(size=10, color="#333"), bgcolor="rgba(255,255,255,.75)")
    fig.add_scatter(x=xs, y=ys, mode="markers+text", text=[str(i) for i in range(len(xs))], textposition="top center",
                    marker=dict(size=8, color="white", line=dict(color=BLUE, width=2)), showlegend=False, name="Nút")
    sx = [inp.nodes[s[0]][0] for s in inp.supports]; sy = [inp.nodes[s[0]][1] for s in inp.supports]
    fig.add_scatter(x=sx, y=sy, mode="markers", marker=dict(symbol="triangle-up", size=14, color=RED),
                    showlegend=False, name="Gối")
    for node, fx, fy, mz in inp.loads:
        px, py = inp.nodes[node]
        mag = math.hypot(fx, fy)
        if mag > 0:
            L0 = 0.12 * span
            fig.add_annotation(x=px, y=py, ax=px - fx / mag * L0, ay=py - fy / mag * L0, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3, arrowcolor=GREEN, arrowwidth=2,
                               text=f"{mag:.0f} kN", font=dict(color=GREEN, size=10))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    pad = 0.12 * span
    fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad]); fig.update_yaxes(range=[min(ys) - pad, max(ys) + 2 * pad])
    return fig_layout(fig, title, 520)


def plot_forces(res, kind):
    inp, model, opt = res.inp, res.model, res.opt
    xs = [p[0] for p in inp.nodes]; ys = [p[1] for p in inp.nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    fig = go.Figure()
    for k, (i, j) in enumerate(inp.members):
        fig.add_scatter(x=[inp.nodes[i][0], inp.nodes[j][0]], y=[inp.nodes[i][1], inp.nodes[j][1]], mode="lines",
                        line=dict(color="#999", width=2), showlegend=False, hoverinfo="skip")
    if inp.mode == "truss":
        for k, (i, j) in enumerate(inp.members):
            Nk = float(opt.N[k])
            col = RED if Nk > 0 else BLUE
            fig.add_scatter(x=[inp.nodes[i][0], inp.nodes[j][0]], y=[inp.nodes[i][1], inp.nodes[j][1]], mode="lines",
                            line=dict(color=col, width=2 + 8 * abs(Nk) / max(np.max(np.abs(opt.N)), 1e-12)),
                            showlegend=False, hoverinfo="text", text=f"Thanh {k}: N = {Nk:.2f} kN")
            fig.add_annotation(x=(inp.nodes[i][0] + inp.nodes[j][0]) / 2, y=(inp.nodes[i][1] + inp.nodes[j][1]) / 2,
                               text=f"{Nk:.1f}", showarrow=False, bgcolor="rgba(255,255,255,.8)", font=dict(size=10, color=col))
        title = "Lực dọc N (kN) — đỏ: kéo, xanh: nén"
    else:
        diag = model.element_diagram(opt.U, opt.A, npts=15)
        key = {"N": "N", "V": "V", "M": "M"}[kind]
        mx = max(float(np.max(np.abs(d[key]))) for d in diag) or 1.0
        sc = 0.10 * span / mx
        for e, d in enumerate(diag):
            nx, ny = -model.s[e], model.c[e]
            ox = d["x"] + nx * d[key] * sc; oy = d["y"] + ny * d[key] * sc
            fig.add_scatter(x=np.concatenate([d["x"], ox[::-1]]), y=np.concatenate([d["y"], oy[::-1]]), fill="toself",
                            fillcolor="rgba(214,39,40,.25)", line=dict(color=RED, width=1), showlegend=False,
                            hoverinfo="skip")
            kk = int(np.argmax(np.abs(d[key])))
            if abs(d[key][kk]) > 0.05 * mx:
                fig.add_annotation(x=ox[kk], y=oy[kk], text=f"{d[key][kk]:.1f}", showarrow=False, font=dict(size=9))
        title = {"N": "Lực dọc N (kN)", "V": "Lực cắt V (kN)", "M": "Mômen M (kNm) — vẽ về phía trục y cục bộ"}[kind]
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig_layout(fig, title, 480)


def tab_struct():
    _init_struct_state()
    st.header("② Toàn bộ khung / dàn kết cấu")
    tok = st.session_state["s_token"]

    top = st.columns([2, 1.3, 3])
    pre = top[0].selectbox("Mẫu có sẵn", list(PRESETS), key="s_preset")
    top[1].write(""); top[1].write("")
    top[1].button("📥 Nạp mẫu", on_click=lambda: _load_preset(st.session_state["s_preset"]), key="s_loadbtn")
    mode_label = top[2].radio("Mô hình liên kết nút", ["Dàn (nút khớp)", "Khung phẳng (nút cứng)"], horizontal=True,
                              key="s_mode")
    is_frame = mode_label.startswith("Khung")

    st.markdown("**Vẽ kết cấu** — kéo chuột để vẽ thanh; chế độ 🔒 bấm vào nút để đổi gối (ngàm → khớp → con lăn → bỏ).")
    geo = st.session_state["geo"]
    val = _canvas_component()(nodes=geo["nodes"], elements=geo["elements"], supports=geo["supports"], snap=0.5,
                              reset_token=tok, height=460, key=f"dop_canvas_{tok}", default=None)
    if val:
        geo = dict(nodes=val["nodes"], elements=val["elements"], supports=val["supports"])
        st.session_state["geo"] = geo
    st.caption(f"{len(geo['nodes'])} nút · {len(geo['elements'])} thanh · {len(geo['supports'])} gối")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Tải tập trung tại nút (kN)")
        nodal_df = st.data_editor(st.session_state["nodal0"], num_rows="dynamic", key=f"s_nodal_{tok}", hide_index=True,
                                  width="stretch")
    with c2:
        st.caption("Tải phân bố đứng trên thanh (kN/m theo chiều dài thanh)")
        udl_df = st.data_editor(st.session_state["udl0"], num_rows="dynamic", key=f"s_udl_{tok}", hide_index=True,
                                width="stretch")
    with c3:
        st.caption("Ràng buộc chuyển vị (mm)")
        lim_df = st.data_editor(st.session_state["lim0"], num_rows="dynamic", key=f"s_lim_{tok}", hide_index=True,
                                width="stretch",
                                column_config={"Phương": st.column_config.SelectboxColumn(options=["ux", "uy"], required=True)})

    with st.expander("⚙️ Vật liệu · họ tiết diện · ổn định", expanded=False):
        a, b, c = st.columns(3)
        E = a.number_input("E (GPa)", 1.0, 400.0, STEEL_E / 1e6, 5.0, key="s_E") * 1e6
        sig = b.number_input("[σ] (MPa)", 10.0, 1000.0, STEEL_SIGMA / 1e3, 10.0, key="s_sig") * 1e3
        rho = c.number_input("ρg (kN/m³)", 1.0, 100.0, STEEL_RHO, 0.5, key="s_rho")
        fam = family_selector("s", default="Ống tròn mỏng")
        a, b, c, d = st.columns(4)
        amin = a.number_input("A_min (cm²)", 0.01, 500.0, 2.0, 0.5, key="s_amin") * 1e-4
        amax = b.number_input("A_max (cm²)", 1.0, 20000.0, 500.0, 10.0, key="s_amax") * 1e-4
        buck = c.checkbox("Ổn định Euler khi nén", True, key="s_buck")
        keff = d.number_input("Hệ số chiều dài k", 0.5, 2.5, 1.0, 0.1, key="s_k")
        fs = a.number_input("FS ổn định", 1.0, 3.0, 1.0, 0.1, key="s_fs")
        st.session_state.setdefault("s_seg", 4)      # giá trị gán qua session_state, không truyền value= (tránh cảnh báo)
        seg = b.number_input("Số đoạn chia / thanh (khung)", 1, 8, step=1, key="s_seg",
                             help="Chia nhỏ để A thay đổi dọc thanh (vút thon trong khung).")

    if st.button("🚀 Tối ưu toàn kết cấu", type="primary", key="s_go"):
        try:
            inp = build_struct_input(geo, mode_label, fam, E, sig, rho, amin, amax, buck, keff, fs, int(seg),
                                     nodal_df, udl_df, lim_df)
            with st.spinner("Đang tối ưu (FEM + OC + KKT)..."):
                st.session_state["s_res"] = optimize_frame(inp)
        except Exception as ex:
            st.session_state.pop("s_res", None)
            st.error(f"Lỗi: {ex}")

    res = st.session_state.get("s_res")
    if res is None:
        st.plotly_chart(_preview_fig(geo, mode_label), width="stretch")
        return
    o = res.opt
    metrics([
        ("Trọng lượng tối ưu", f"{res.weight:,.2f} kN"),
        ("Tiết diện đều (mốc)", f"{res.weight_uniform:,.2f} kN"),
        ("Tiết kiệm", f"{res.saving * 100:.1f}%", "metric-good"),
        ("U lớn nhất", f"{float(np.max(o.util)):.3f}"),
        ("Hội tụ / thoả", f"{'✔' if o.converged else '…'} / {'✔' if o.feasible else '✘'}",
         "metric-good" if o.feasible else "metric-bad"),
        ("Sai số KKT (E–L)", f"{o.EL_cv:.1%}" if o.EL_cv is not None else "—"),
    ])
    t1, t2, t3, t4 = st.tabs(["Tiết diện tối ưu", "Nội lực", "Bảng thanh", "Hội tụ"])
    with t1:
        st.plotly_chart(plot_structure(res.inp, area_member=res.A_member, title="Diện tích A (cm²) — độ dày/màu ∝ A"),
                        width="stretch")
    with t2:
        kinds = ["N"] if res.inp.mode == "truss" else ["M", "N", "V"]
        k = st.radio("Biểu đồ", kinds, horizontal=True, key="s_diag") if len(kinds) > 1 else kinds[0]
        st.plotly_chart(plot_forces(res, k), width="stretch")
    with t3:
        rows = []
        for m_, (i, j) in enumerate(res.inp.members):
            sel = res.member_of_el == m_
            Nk = float(o.N[sel][np.argmax(np.abs(o.N[sel]))])
            rows.append({"Thanh": m_, "i-j": f"{i}-{j}", "L (m)": float(np.sum(res.model.L[sel])),
                         "A (cm²)": res.A_member[m_] * 1e4, "N (kN)": Nk, "Mmax (kNm)": float(np.max(o.M[sel])),
                         "U": float(np.max(o.util[sel]))})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        if res.crosscheck is not None:
            st.caption(f"Đối chiếu fem_core.solve_plane_frame: sai khác chuyển vị tương đối = {res.crosscheck['rel_diff']:.1e}")
    with t4:
        h = pd.DataFrame(o.history)
        fg = make_subplots(rows=1, cols=2, subplot_titles=("Trọng lượng (kN)", "Utilization / tỷ số chuyển vị lớn nhất"))
        fg.add_scatter(y=h["weight"], row=1, col=1, name="W", line=dict(color=BLUE))
        fg.add_scatter(y=h["max_util"], row=1, col=2, name="U max", line=dict(color=RED))
        fg.add_scatter(y=h["max_disp_ratio"], row=1, col=2, name="δ/δ_lim", line=dict(color=ORANGE))
        st.plotly_chart(fig_layout(fg, "", 360), width="stretch")
    report_block(res.report, "khung_toi_uu", "Thuyết minh — Tối ưu kết cấu khung/dàn")


def _preview_fig(geo, mode_label):
    inp = StructInput(nodes=[(n["x"], n["y"]) for n in geo["nodes"]], members=[(e["i"], e["j"]) for e in geo["elements"]],
                      supports=[(s["node"], s["ux"], s["uy"], s["rz"]) for s in geo["supports"]], loads=[], member_q=[],
                      mode="frame", fam=make_family("circle"))
    return plot_structure(inp, title="Xem trước hình học (đánh số nút/thanh)")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    inject_css()
    st.title("🦈 Đớp Tool")
    st.caption("Tối ưu hoá kết cấu dàn/khung thép bằng Phép tính Biến phân (Euler–Lagrange) + FEM")
    t0, t1, t2, t3 = st.tabs(["📘 Lý thuyết", "① Cột", "① Dầm vút thon", "② Toàn khung / dàn"])
    with t0:
        tab_theory()
    with t1:
        tab_column()
    with t2:
        tab_beam()
    with t3:
        tab_struct()


main()
