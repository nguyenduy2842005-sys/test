"""
legacy_plots.py — các hàm vẽ biểu đồ được TRÍCH NGUYÊN VĂN từ trang web cũ (web_app_v2.py).

Không viết lại: toàn bộ thân hàm giữ đúng như bản gốc để giao diện/ký hiệu (gối, tải, SFD/BMD, nội lực khung)
đồng nhất với trang cũ. Chỉ khác: (1) đưa các hằng số màu vào đầu file, (2) màu khung nền và màu chữ tiêu đề
dùng biến để hợp cả theme sáng và tối (xem BASE_FRAME_COLOR / TITLE_COLOR).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PLOT_HEIGHT = 315

COLOR_SFD   = "#0b5fff"
COLOR_BMD   = "#ff2b2b"
COLOR_ELAST = "#ff8800"
COLOR_AXIAL = "#9b27af"
COLOR_BEAM  = "#7a7f85"
COLOR_SUP   = "#ef1d14"

# Thay đổi theo theme (được web_app đặt): nền khung trong biểu đồ nội lực.
BASE_FRAME_COLOR = "#eef3fb"

# plot_sfd_single / plot_bmd_single chỉ cần đối tượng có .x .shear .moment (duck-typing)
BeamResult = object


def _padded_range(values: np.ndarray, pad_frac: float = 0.18, min_span: float = 1.0) -> tuple[float, float]:
    """Tính khoảng [min,max] có đệm biên hợp lý cho trục y, dựa trên dữ liệu thật.
    Đảm bảo trục được KHÓA Ở MỘT TỶ LỆ CỐ ĐỊNH (không tự ý co giãn theo từng điểm hover),
    nhưng vẫn hiển thị trọn vẹn toàn bộ biểu đồ."""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    if span < min_span:
        center = (vmax + vmin) / 2
        vmin, vmax = center - min_span / 2, center + min_span / 2
        span = vmax - vmin
    pad = span * pad_frac
    return (vmin - pad, vmax + pad)

def base_figure(title, length, y_title="", reverse_y=None):
    margin_x = max(length * 0.06, 0.5)

    fig = go.Figure()

    # Tự động nhận diện biểu đồ mô-men qua tiêu đề nếu không truyền tường minh reverse_y
    if reverse_y is None:
        title_lower = title.lower()
        reverse_y = any(kw in title_lower for kw in ["mô-men", "momen", "moment", "bmd"])

    # Range mặc định của bạn gốc là [-1.5, 1.2] (vmin = -1.5, vmax = 1.2)
    if reverse_y:
        y_range_config = [1.2, -1.5]  # [vmax, vmin] -> Ép trục Y đảo ngược cố định (Dương hướng xuống)
    else:
        y_range_config = [-1.5, 1.2]  # [vmin, vmax] -> Trục Y xuôi thông thường (Dương hướng lên)

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=14)
        ),
        height=315,
        margin=dict(l=55, r=20, t=60, b=75),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(
            title="x (m)",
            range=[-margin_x, length + margin_x],
            zeroline=True,
            mirror=True,
            showgrid=True,
            linecolor="gray",
            gridcolor="rgba(128,128,128,0.2)"
        ),
        yaxis=dict(
            title=y_title,
            range=y_range_config,  # <--- Áp dụng cấu hình range đã chuẩn hóa hình học
            zeroline=True,
            mirror=True,
            showgrid=True,
            linecolor="gray",
            gridcolor="rgba(128,128,128,0.2)"
        )
    )
    return fig

def synced_figure(title, length, y_range=(-1.5, 1.2), y_title="", reverse_y=None):
    margin_x = max(length * 0.06, 0.5)

    fig = go.Figure()

    # Tự động nhận diện biểu đồ mô-men qua tiêu đề nếu không truyền tường minh reverse_y
    if reverse_y is None:
        title_lower = title.lower()
        reverse_y = any(kw in title_lower for kw in ["mô-men", "momen", "moment", "bmd"])

    # Xử lý dải giá trị y_range dựa trên việc có đảo trục hay không
    if y_range is not None:
        vmin = min(y_range)
        vmax = max(y_range)
        if reverse_y:
            actual_y_range = [vmax, vmin]  # Số lớn xếp trước số nhỏ -> Ép cả Web lẫn Kaleido lật ngược trục Y 100%
        else:
            actual_y_range = [vmin, vmax]  # Số nhỏ xếp trước số lớn -> Trục xuôi toán học thông thường
    else:
        actual_y_range = None

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=15)
        ),
        height=PLOT_HEIGHT,
        margin=dict(l=55, r=25, t=60, b=65),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(
            title="x (m)",
            range=[-margin_x, length + margin_x],
            zeroline=True,
            mirror=True,
            scaleanchor=None
        ),
        yaxis=dict(
            title=y_title,
            range=actual_y_range,  # <--- Sử dụng range cấu hình đảo tường minh ở đây
            zeroline=True,
            mirror=True
        )
    )

    fig.update_xaxes(
        showgrid=True,
        linecolor="gray",
        gridcolor="rgba(128,128,128,0.2)"
    )
    fig.update_yaxes(
        showgrid=True,
        linecolor="gray",
        gridcolor="rgba(128,128,128,0.2)"
    )
    return fig

def _cb_draw_base_beam_and_supports(total_L, span_lengths, support_kinds) -> go.Figure:
    """Hàm dựng khung dầm và gối đỡ nền đồng bộ tỷ lệ hiển thị"""

    fig = go.Figure()

    # khoảng trống hai đầu để không cắt gối
    margin_x = max(total_L * 0.06, 0.5)

    # 1. Layout
    fig.update_layout(
        title=dict(
            x=0.5,
            xanchor="center",
            font=dict(size=14)
        ),

        height=315,

        margin=dict(
            l=55,
            r=20,
            t=60,
            b=75
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,

        xaxis=dict(
            title="x (m)",
            range=[-margin_x, total_L + margin_x],
            showgrid=True,
            linecolor="gray",
            gridcolor="rgba(128,128,128,0.15)"
        )
    )


    fig.update_yaxes(
        range=[-1.5, 1.2],
        fixedrange=True,
        showgrid=False,
        linecolor="gray",
        showticklabels=False,
        title=""
    )


    # 2. Thanh dầm
    fig.add_trace(
        go.Scatter(
            x=[0, total_L],
            y=[0, 0],
            mode="lines",
            line={
                "color": COLOR_BEAM,
                "width": 8
            },
            hoverinfo="skip"
        )
    )


    # Nhãn nhịp
    x_acc = 0.0

    for i, Ls in enumerate(span_lengths):
        mid = x_acc + Ls / 2

        fig.add_annotation(
            x=mid,
            y=0.2,
            text=f"L{i+1}={Ls:.1f}m",
            showarrow=False,
            font={
                "size":10,
                "color":"#8a8f98"
            }
        )

        x_acc += Ls



    # 3. Gối
    node_xs = [0.0] + list(np.cumsum(span_lengths))

    support_size = max(total_L * 0.035, 0.25)


    for i, kind in enumerate(support_kinds):

        xp = node_xs[i]


        # gối pin
        if kind == "pin":

            fig.add_trace(
                go.Scatter(
                    x=[
                        xp,
                        xp + support_size,
                        xp - support_size,
                        xp
                    ],

                    y=[
                        0,
                        -0.37,
                        -0.37,
                        0
                    ],

                    fill="toself",
                    mode="lines",
                    line={
                        "color":COLOR_SUP,
                        "width":1.5
                    },
                    fillcolor=COLOR_SUP,
                    hoverinfo="skip"
                )
            )


        # gối con lăn
        elif kind == "roller":

            y_top = -0.08
            y_bot = -0.36
            y_floor = -0.44


            fig.add_trace(
                go.Scatter(
                    x=[xp],
                    y=[y_top],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=7,
                        color=COLOR_SUP
                    ),
                    hoverinfo="skip"
                )
            )


            fig.add_trace(
                go.Scatter(
                    x=[xp],
                    y=[y_bot],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=7,
                        color=COLOR_SUP
                    ),
                    hoverinfo="skip"
                )
            )


            fig.add_trace(
                go.Scatter(
                    x=[xp,xp],
                    y=[y_top,y_bot],
                    mode="lines",
                    line=dict(
                        color=COLOR_SUP,
                        width=1.5
                    ),
                    hoverinfo="skip"
                )
            )


            fig.add_trace(
                go.Scatter(
                    x=[
                        xp-support_size,
                        xp+support_size
                    ],
                    y=[
                        y_floor,
                        y_floor
                    ],
                    mode="lines",
                    line=dict(
                        color=COLOR_SUP,
                        width=2
                    ),
                    hoverinfo="skip"
                )
            )



        # ngàm
        elif kind == "fixed":

            fig.add_shape(
                type="rect",
                x0=xp-total_L/80,
                x1=xp+total_L/80,
                y0=-0.42,
                y1=0.42,
                fillcolor=COLOR_SUP,
                line={"color":COLOR_SUP}
            )


        fig.add_annotation(
            x=xp,
            y=-0.68,
            text=f"N{i}",
            showarrow=False,
            font={
                "size":10,
                "color":COLOR_SUP
            }
        )


    return fig

def _pf_support_type(row) -> str:
    """
    Xác định loại gối tựa từ 3 cờ ràng buộc (ux, uy, rz) của bảng Supports,
    dùng chung quy ước với canvas vẽ khung (xem hàm JS boolToType):
      - FIXED : ux, uy, rz đều bị ràng buộc      -> ngàm cứng
      - PIN   : ux, uy bị ràng buộc, rz tự do    -> khớp cố định
      - ROLX  : chỉ uy bị ràng buộc (ux tự do)   -> gối di động, trượt ngang
      - ROLY  : chỉ ux bị ràng buộc (uy tự do)   -> gối di động, trượt đứng
    """
    ux = bool(row.get("ux", False))
    uy = bool(row.get("uy", False))
    rz = bool(row.get("rz", False))

    if ux and uy and rz:
        return "FIXED"
    if ux and uy and not rz:
        return "PIN"
    if not ux and uy:
        return "ROLX"
    if ux and not uy:
        return "ROLY"
    return "PIN"

def draw_support_pf(fig: go.Figure, x: float, y: float, sup_type: str, size: float) -> None:
    """
    Vẽ ký hiệu gối tựa tại node (x, y) của khung phẳng, ĐỒNG BỘ phong cách
    vẽ với các tab Dầm đơn / Dầm liên tục (tam giác đặc cho khớp, chấm
    tròn + gạch chân cho gối di động, hình chữ nhật đặc có gạch hatch cho
    ngàm cứng) — dùng chung màu COLOR_SUP.
    """
    h = size * 1.15   # chiều cao tam giác / khoảng lùi gối
    w = size * 0.95   # nửa bề rộng đáy tam giác
    floor = size * 1.5

    if sup_type == "FIXED":
        fig.add_shape(
            type="rect",
            x0=x - size, x1=x + size,
            y0=y - size * 0.4, y1=y,
            fillcolor=COLOR_SUP, line={"color": COLOR_SUP},
        )
        for k in range(-2, 3):
            xk = x + k * size * 0.4
            fig.add_trace(go.Scatter(
                x=[xk, xk - size * 0.25], y=[y - size * 0.4, y - size * 0.8],
                mode="lines", line=dict(color=COLOR_SUP, width=1.2),
                hoverinfo="skip", showlegend=False,
            ))

    elif sup_type == "PIN":
        fig.add_trace(go.Scatter(
            x=[x, x + w, x - w, x], y=[y, y - h, y - h, y],
            fill="toself", mode="lines",
            line={"color": COLOR_SUP, "width": 1.5}, fillcolor=COLOR_SUP,
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[x - w * 1.3, x + w * 1.3], y=[y - h, y - h],
            mode="lines", line=dict(color=COLOR_SUP, width=2),
            hoverinfo="skip", showlegend=False,
        ))

    elif sup_type == "ROLX":  # ngăn chuyển vị đứng, tự do trượt ngang
        y_top, y_bot, y_floor = y - size * 0.25, y - size * 1.05, y - floor
        fig.add_trace(go.Scatter(
            x=[x], y=[y_top], mode="markers",
            marker=dict(symbol="circle", size=7, color=COLOR_SUP),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y_bot], mode="markers",
            marker=dict(symbol="circle", size=7, color=COLOR_SUP),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[x, x], y=[y_top, y_bot], mode="lines",
            line=dict(color=COLOR_SUP, width=1.5),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[x - w, x + w], y=[y_floor, y_floor], mode="lines",
            line=dict(color=COLOR_SUP, width=2),
            hoverinfo="skip", showlegend=False,
        ))

    else:  # ROLY: ngăn chuyển vị ngang, tự do trượt đứng (xoay 90° so với ROLX)
        x_top, x_bot, x_floor = x - size * 0.25, x - size * 1.05, x - floor
        fig.add_trace(go.Scatter(
            x=[x_top], y=[y], mode="markers",
            marker=dict(symbol="circle", size=7, color=COLOR_SUP),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[x_bot], y=[y], mode="markers",
            marker=dict(symbol="circle", size=7, color=COLOR_SUP),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[x_top, x_bot], y=[y, y], mode="lines",
            line=dict(color=COLOR_SUP, width=1.5),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[x_floor, x_floor], y=[y - w, y + w], mode="lines",
            line=dict(color=COLOR_SUP, width=2),
            hoverinfo="skip", showlegend=False,
        ))

def _pf_geometry_plot(
    df_nodes: pd.DataFrame,
    df_el: pd.DataFrame,
    df_sup: pd.DataFrame,
    result_pf=None,
    df_nload: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Vẽ hình học khung phẳng 2D.

    Parameters
    ----------
    df_nodes : DataFrame
        Bảng Nodes gồm:
        - x (m)
        - y (m)

    df_el : DataFrame
        Bảng Elements gồm:
        - i
        - j

    df_sup : DataFrame
        Bảng Supports gồm:
        - node
        - ux
        - uy
        - rz

    result_pf : PlaneFrameResult, optional
        Kết quả FEM. Hiện tại dùng để giữ tương thích
        với giao diện và có thể mở rộng sau này.

    df_nload : DataFrame, optional
        Bảng Node Loads gồm: node, Fx (kN), Fy (kN), Mz (kNm).
        Dùng để vẽ mũi tên lực / mô-men gán vào node.
    """

    fig = go.Figure()

    # ==========================================================
    # 1. KIỂM TRA DỮ LIỆU NODES
    # ==========================================================

    if df_nodes is None or df_nodes.empty:
        fig.update_layout(
            title="Plane Frame Geometry",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            template="plotly_white",
        )
        return fig

    nodes = df_nodes.copy()

    # Đảm bảo tên cột tồn tại
    required_node_cols = ["x (m)", "y (m)"]

    if not all(c in nodes.columns for c in required_node_cols):
        fig.update_layout(
            title="Plane Frame Geometry — Invalid Node Data",
            template="plotly_white",
        )
        return fig

    # Ép kiểu số
    nodes["x (m)"] = pd.to_numeric(
        nodes["x (m)"],
        errors="coerce"
    )

    nodes["y (m)"] = pd.to_numeric(
        nodes["y (m)"],
        errors="coerce"
    )

    nodes = nodes.dropna(
        subset=["x (m)", "y (m)"]
    ).reset_index(drop=True)

    # ==========================================================
    # 2. VẼ ELEMENTS
    # ==========================================================

    if df_el is not None and not df_el.empty:

        elements = df_el.copy()

        if "i" in elements.columns and "j" in elements.columns:

            for _, row in elements.iterrows():

                try:
                    i = int(row["i"])
                    j = int(row["j"])

                    if (
                        i < 0
                        or j < 0
                        or i >= len(nodes)
                        or j >= len(nodes)
                    ):
                        continue

                    xi = float(nodes.loc[i, "x (m)"])
                    yi = float(nodes.loc[i, "y (m)"])

                    xj = float(nodes.loc[j, "x (m)"])
                    yj = float(nodes.loc[j, "y (m)"])

                    fig.add_trace(
                        go.Scatter(
                            x=[xi, xj],
                            y=[yi, yj],
                            mode="lines",
                            line=dict(
                                color="#1f77b4",
                                width=4,
                            ),
                            hovertemplate=(
                                f"Element: {i} → {j}"
                                "<br>X: %{x:.3f} m"
                                "<br>Y: %{y:.3f} m"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                ):
                    continue

    # ==========================================================
    # 2b. VẼ TẢI PHÂN BỐ ĐỀU (UDL) TRÊN TỪNG PHẦN TỬ
    # ==========================================================
    # FIX: UDL đã được gán ở bảng Elements (cột udl_local) và đã tính đúng
    # trong bước Solve, nhưng trước đây KHÔNG BAO GIỜ được vẽ lên hình học
    # -> người dùng gán tải mà không thấy nó nằm ở đâu. Bổ sung dải tải +
    # mũi tên dọc theo phần tử, đồng bộ màu xanh lá với quy ước UDL của
    # 2 tab Dầm (Single Beam / Continuous Beam).
    #
    # Quy ước: q dương theo chiều trục y cục bộ của phần tử — trục y cục bộ
    # được xác định bằng cách quay trục x cục bộ (hướng từ node i -> j)
    # 90° ngược chiều kim đồng hồ (quy tắc bàn tay phải).

    if (
        df_el is not None and not df_el.empty
        and "i" in df_el.columns and "j" in df_el.columns
        and "udl_local" in df_el.columns
    ):

        x_span_u = float(nodes["x (m)"].max() - nodes["x (m)"].min()) if len(nodes) else 0.0
        y_span_u = float(nodes["y (m)"].max() - nodes["y (m)"].min()) if len(nodes) else 0.0
        L_off = max(max(x_span_u, y_span_u) * 0.12, 0.35)

        for _, row in df_el.iterrows():
            try:
                i = int(row["i"])
                j = int(row["j"])

                q = row.get("udl_local")
                q = float(q) if q not in (None, "") and pd.notna(q) else 0.0
                if abs(q) < 1e-9:
                    continue

                if i < 0 or j < 0 or i >= len(nodes) or j >= len(nodes):
                    continue

                xi, yi = float(nodes.loc[i, "x (m)"]), float(nodes.loc[i, "y (m)"])
                xj, yj = float(nodes.loc[j, "x (m)"]), float(nodes.loc[j, "y (m)"])

                dx, dy = xj - xi, yj - yi
                L = math.hypot(dx, dy)
                if L < 1e-9:
                    continue

                tx, ty = dx / L, dy / L        # trục x cục bộ (i -> j)
                nx, ny = -ty, tx                # trục y cục bộ (quay 90° ngược kim đồng hồ — khớp T trong fem_core._rotation_matrix)
                sign = 1.0 if q > 0 else -1.0
                # Mũi tên phải TRỎ THEO chiều lực (kết thúc tại thanh), nên điểm gốc (đuôi mũi tên)
                # nằm ở phía NGƯỢC với chiều +local_y*sign(q); dải tải tô mờ cũng đặt cùng phía đuôi mũi tên.
                ox, oy = -nx * L_off * sign, -ny * L_off * sign

                # Dải tải phân bố (vùng tô mờ)
                fig.add_trace(
                    go.Scatter(
                        x=[xi, xj, xj + ox, xi + ox, xi],
                        y=[yi, yj, yj + oy, yi + oy, yi],
                        fill="toself",
                        mode="lines",
                        line=dict(color="#28a745", width=1),
                        fillcolor="rgba(40,167,69,0.15)",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                # Mũi tên phân bố dọc theo phần tử, hướng vào thanh
                n_arrows = max(3, int(L / 0.9))
                for k in range(n_arrows + 1):
                    tt = k / n_arrows
                    px_, py_ = xi + tx * L * tt, yi + ty * L * tt
                    fig.add_annotation(
                        x=px_, y=py_,
                        ax=px_ + ox, ay=py_ + oy,
                        axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=2, arrowsize=0.9, arrowwidth=1.6,
                        arrowcolor="#28a745",
                        text="",
                    )

                # Nhãn giá trị q tại điểm giữa
                xm, ym = (xi + xj) / 2 + ox, (yi + yj) / 2 + oy
                fig.add_annotation(
                    x=xm, y=ym,
                    text=f"q={q:g} kN/m",
                    showarrow=False,
                    font=dict(size=11, color="#168f2c"),
                    bgcolor="rgba(255,255,255,0.85)",
                )

            except (TypeError, ValueError, KeyError, IndexError):
                continue

    # ==========================================================
    # 3. VẼ NODES
    # ==========================================================

    fig.add_trace(
        go.Scatter(
            x=nodes["x (m)"],
            y=nodes["y (m)"],
            mode="markers+text",
            text=[
                f"N{i}"
                for i in range(len(nodes))
            ],
            textposition="top center",
            textfont=dict(size=12, color="#ffffff"),
            marker=dict(
                size=10,
                color="#ffffff",
                line=dict(
                    color="#1f77b4",
                    width=2,
                ),
            ),
            name="Nodes",
            hovertemplate=(
                "Node: %{text}"
                "<br>X: %{x:.3f} m"
                "<br>Y: %{y:.3f} m"
                "<extra></extra>"
            ),
        )
    )

    # ==========================================================
    # 4. VẼ GỐI TỰA
    # ==========================================================

    if df_sup is not None and not df_sup.empty:

        supports = df_sup.copy()

        if "node" in supports.columns:

            # Kích thước ký hiệu gối tỷ lệ theo bao hình học của khung,
            # cùng cách tính với _cb_draw_base_beam_and_supports để đồng bộ
            x_span_sup = float(nodes["x (m)"].max() - nodes["x (m)"].min()) if len(nodes) else 0.0
            y_span_sup = float(nodes["y (m)"].max() - nodes["y (m)"].min()) if len(nodes) else 0.0
            sup_size = max(max(x_span_sup, y_span_sup) * 0.035, 0.25)

            for _, row in supports.iterrows():

                try:
                    node_id = int(row["node"])

                    if (
                        node_id < 0
                        or node_id >= len(nodes)
                    ):
                        continue

                    x = float(
                        nodes.loc[node_id, "x (m)"]
                    )

                    y = float(
                        nodes.loc[node_id, "y (m)"]
                    )

                    sup_type = _pf_support_type(row)
                    draw_support_pf(fig, x, y, sup_type, sup_size)

                except (
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                ):
                    continue

    # ==========================================================
    # 5. ĐÁNH SỐ ELEMENT
    # ==========================================================

    if df_el is not None and not df_el.empty:

        elements = df_el.copy()

        if "i" in elements.columns and "j" in elements.columns:

            for e_id, row in elements.iterrows():

                try:
                    i = int(row["i"])
                    j = int(row["j"])

                    if (
                        i < 0
                        or j < 0
                        or i >= len(nodes)
                        or j >= len(nodes)
                    ):
                        continue

                    xi = float(nodes.loc[i, "x (m)"])
                    yi = float(nodes.loc[i, "y (m)"])

                    xj = float(nodes.loc[j, "x (m)"])
                    yj = float(nodes.loc[j, "y (m)"])

                    xm = (xi + xj) / 2
                    ym = (yi + yj) / 2

                    fig.add_annotation(
                        x=xm,
                        y=ym,
                        text=f"E{e_id}",
                        showarrow=False,
                        font=dict(
                            size=12,
                            color="#111111",
                        ),
                        bgcolor="rgba(255,255,255,0.75)",
                    )

                except (
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                ):
                    continue

    # ==========================================================
    # 5b. VE TAI TRONG GAN VAO NODE (Fx, Fy, Mz)
    # ==========================================================
    # FIX: truoc day ham nay khong nhan df_nload nen tai trong da gan
    # o tab "Node Loads" khong bao gio duoc ve len hinh, du da tinh
    # dung o buoc Solve.

    if df_nload is not None and not df_nload.empty and "node" in df_nload.columns:

        x_span = float(nodes["x (m)"].max() - nodes["x (m)"].min()) if len(nodes) else 0.0
        y_span = float(nodes["y (m)"].max() - nodes["y (m)"].min()) if len(nodes) else 0.0
        L_arrow = max(x_span, y_span, 1.0) * 0.18

        for _, row in df_nload.iterrows():

            try:
                node_id = row.get("node")
                if node_id is None or pd.isna(node_id):
                    continue
                node_id = int(node_id)

                if node_id < 0 or node_id >= len(nodes):
                    continue

                x0 = float(nodes.loc[node_id, "x (m)"])
                y0 = float(nodes.loc[node_id, "y (m)"])

                fx = row.get("Fx (kN)")
                fy = row.get("Fy (kN)")
                mz = row.get("Mz (kNm)")

                fx = float(fx) if fx not in (None, "") and pd.notna(fx) else 0.0
                fy = float(fy) if fy not in (None, "") and pd.notna(fy) else 0.0
                mz = float(mz) if mz not in (None, "") and pd.notna(mz) else 0.0

                if fx != 0.0:
                    sign = 1.0 if fx > 0 else -1.0
                    fig.add_annotation(
                        x=x0, y=y0,
                        ax=x0 - sign * L_arrow, ay=y0,
                        axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=3, arrowsize=1.1, arrowwidth=2.2,
                        arrowcolor=COLOR_SFD,
                        text="",
                    )
                    fig.add_annotation(
                        x=x0 - sign * L_arrow, y=y0,
                        text=f"Fx={fx:g} kN",
                        showarrow=False,
                        font=dict(size=11, color=COLOR_SFD),
                        bgcolor="rgba(255,255,255,0.8)",
                        yshift=12,
                    )

                if fy != 0.0:
                    sign = 1.0 if fy > 0 else -1.0
                    fig.add_annotation(
                        x=x0, y=y0,
                        ax=x0, ay=y0 - sign * L_arrow,
                        axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=3, arrowsize=1.1, arrowwidth=2.2,
                        arrowcolor=COLOR_SFD,
                        text="",
                    )
                    fig.add_annotation(
                        x=x0, y=y0 - sign * L_arrow,
                        text=f"Fy={fy:g} kN",
                        showarrow=False,
                        font=dict(size=11, color=COLOR_SFD),
                        bgcolor="rgba(255,255,255,0.8)",
                        xshift=32,
                    )

                if mz != 0.0:
                    symbol = "↻" if mz < 0 else "↺"  # chieu kim / nguoc kim theo dau quy uoc
                    fig.add_annotation(
                        x=x0, y=y0,
                        text=f"{symbol} Mz={mz:g} kNm",
                        showarrow=False,
                        font=dict(size=11, color="#ff2b8a"),
                        bgcolor="rgba(255,255,255,0.8)",
                        xshift=-6,
                        yshift=-18,
                    )

            except (TypeError, ValueError, KeyError, IndexError):
                continue

    # ==========================================================
    # 6. LAYOUT
    # ==========================================================

    fig.update_layout(
        title="Plane Frame Geometry",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        template="plotly_white",
        height=500,
        margin=dict(
            l=50,
            r=30,
            t=60,
            b=50,
        ),
        xaxis=dict(
            zeroline=True,
            showgrid=True,
        ),
        yaxis=dict(
            zeroline=True,
            showgrid=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig

def _pf_frame_diagram(
    df_nodes: pd.DataFrame,
    df_el: pd.DataFrame,
    df_sup: pd.DataFrame,
    result_pf,
    diagram_type: str,
    title: str,
) -> go.Figure:
    """
    Ve noi luc (M/V/N) TRUC TIEP TREN HINH KHUNG PHANG - dung phong cach
    phan mem ket cau chuyen dung (SAP2000/ETABS/RDM...): duong bieu do
    duoc ve vuong goc voi truc thanh, lech sang mot ben theo dau gia tri,
    co gach hatch va ghi tri so tai 2 dau + diem cuc tri, thay vi ve theo
    truc hoanh/tung do rieng (khong con "Normalized Element Coordinate").

    Quy uoc: chieu lech (ben nao cua thanh) chi phan anh DAU cua gia tri
    noi luc theo he truc cuc bo cua phan tu trong lien ket voi solver hien
    tai — chua doi chieu theo "tho chiu keo" nhu quy uoc ve tay truyen
    thong. Tri so tuyet doi va vi tri cuc tri la chinh xac tu ket qua FEM.
    """

    fig = go.Figure()

    diagram_type = str(diagram_type).strip().lower()
    attr_map = {
        "moment": ("moment", COLOR_BMD),
        "shear": ("shear", COLOR_SFD),
        "axial": ("axial", "#168f2c"),
    }
    if diagram_type not in attr_map:
        fig.update_layout(title=f"{title} — Loại biểu đồ không hợp lệ", template="plotly_white", height=460)
        return fig
    data_attr, diag_color = attr_map[diagram_type]

    if df_nodes is None or df_nodes.empty:
        fig.update_layout(title=title, template="plotly_white", height=460)
        return fig

    nodes = df_nodes.copy()
    if not all(c in nodes.columns for c in ["x (m)", "y (m)"]):
        fig.update_layout(title=title, template="plotly_white", height=460)
        return fig
    nodes["x (m)"] = pd.to_numeric(nodes["x (m)"], errors="coerce")
    nodes["y (m)"] = pd.to_numeric(nodes["y (m)"], errors="coerce")
    nodes = nodes.dropna(subset=["x (m)", "y (m)"]).reset_index(drop=True)

    # ---------------------------------------------------------------
    # 1. VE KHUNG NEN (net manh, mau xam den) lam khung tham chieu
    # ---------------------------------------------------------------
    if df_el is not None and not df_el.empty and "i" in df_el.columns and "j" in df_el.columns:
        for _, row in df_el.iterrows():
            try:
                i, j = int(row["i"]), int(row["j"])
                if i < 0 or j < 0 or i >= len(nodes) or j >= len(nodes):
                    continue
                xi, yi = float(nodes.loc[i, "x (m)"]), float(nodes.loc[i, "y (m)"])
                xj, yj = float(nodes.loc[j, "x (m)"]), float(nodes.loc[j, "y (m)"])
                fig.add_trace(go.Scatter(
                    x=[xi, xj], y=[yi, yj], mode="lines",
                    line=dict(color=BASE_FRAME_COLOR, width=5),
                    hoverinfo="skip", showlegend=False,
                ))
            except (TypeError, ValueError, KeyError, IndexError):
                continue

    if df_sup is not None and not df_sup.empty and "node" in df_sup.columns:
        # Kích thước ký hiệu gối đồng bộ với _pf_geometry_plot / các tab dầm
        x_span_sup = float(nodes["x (m)"].max() - nodes["x (m)"].min()) if len(nodes) else 0.0
        y_span_sup = float(nodes["y (m)"].max() - nodes["y (m)"].min()) if len(nodes) else 0.0
        sup_size = max(max(x_span_sup, y_span_sup) * 0.035, 0.25)

        for _, row in df_sup.iterrows():
            try:
                node_id = int(row["node"])
                if node_id < 0 or node_id >= len(nodes):
                    continue
                x = float(nodes.loc[node_id, "x (m)"]); y = float(nodes.loc[node_id, "y (m)"])
                sup_type = _pf_support_type(row)
                draw_support_pf(fig, x, y, sup_type, sup_size)
            except (TypeError, ValueError, KeyError, IndexError):
                continue

    if result_pf is None:
        fig.update_layout(
            title=f"{title} — Chưa có kết quả", template="plotly_white", height=460,
            xaxis_title="X (m)", yaxis_title="Y (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        return fig

    element_results = getattr(result_pf, "element_results", None)
    if not element_results:
        fig.update_layout(
            title=f"{title} — Không có dữ liệu", template="plotly_white", height=460,
            xaxis_title="X (m)", yaxis_title="Y (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        return fig

    # ---------------------------------------------------------------
    # 2. TIM HE SO TY LE CHUNG (dung 1 ty le cho toan bo khung, giong
    #    cach cac phan mem ket cau ve dong bo giua cac phan tu)
    # ---------------------------------------------------------------
    all_vals = []
    for er in element_results:
        v = getattr(er, data_attr, None)
        if v is None:
            continue
        try:
            v = np.asarray(v, dtype=float).flatten()
        except Exception:
            continue
        if v.size:
            all_vals.append(v)

    if not all_vals:
        fig.add_annotation(text=f"Không tìm thấy dữ liệu {diagram_type} trong kết quả FEM.",
                            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=13))
        fig.update_layout(title=title, template="plotly_white", height=460)
        return fig

    max_abs = max(float(np.max(np.abs(v))) for v in all_vals)

    x_span = float(nodes["x (m)"].max() - nodes["x (m)"].min()) if len(nodes) else 0.0
    y_span = float(nodes["y (m)"].max() - nodes["y (m)"].min()) if len(nodes) else 0.0
    frame_extent = max(x_span, y_span, 1.0)
    scale = (frame_extent * 0.22 / max_abs) if max_abs > 1e-9 else 0.0

    # ---------------------------------------------------------------
    # 3. VE BIEU DO NOI LUC LECH VUONG GOC TRUC THANH (hatch + duong bao)
    # ---------------------------------------------------------------
    hatch_x, hatch_y = [], []
    ann_seen = set()

    for e_id, er in enumerate(element_results):
        values = getattr(er, data_attr, None)
        xg = getattr(er, "x_coords", None)
        yg = getattr(er, "y_coords", None)
        if values is None or xg is None or yg is None:
            continue
        try:
            values = np.asarray(values, dtype=float).flatten()
            xg = np.asarray(xg, dtype=float).flatten()
            yg = np.asarray(yg, dtype=float).flatten()
        except Exception:
            continue
        n = min(len(values), len(xg), len(yg))
        if n < 2:
            continue
        values, xg, yg = values[:n], xg[:n], yg[:n]

        dx, dy = xg[-1] - xg[0], yg[-1] - yg[0]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        tx, ty = dx / L, dy / L
        nx, ny = -ty, tx  # phap tuyen, quay 90 deg nguoc kim dong ho tu truc doc thanh

        ox = xg + values * scale * nx
        oy = yg + values * scale * ny

        for k in range(n):
            hatch_x.extend([xg[k], ox[k], None])
            hatch_y.extend([yg[k], oy[k], None])

        fig.add_trace(go.Scatter(
            x=ox, y=oy, mode="lines",
            line=dict(color=diag_color, width=2.4),
            name=f"E{e_id}",
            hovertemplate=f"E{e_id}<br>{title} = %{{customdata:.3f}}<extra></extra>",
            customdata=values,
            showlegend=False,
        ))

        # Nhan tri so tai 2 dau phan tu (tranh trung lap qua nhieu tai cung 1 node)
        for k, label_pos in [(0, "start"), (n - 1, "end")]:
            key = (round(xg[k], 4), round(yg[k], 4), round(values[k], 4))
            if key in ann_seen:
                continue
            ann_seen.add(key)
            if abs(values[k]) < 1e-6:
                continue
            fig.add_annotation(
                x=ox[k], y=oy[k], text=f"{values[k]:.3g}",
                showarrow=False, font=dict(size=10, color=diag_color),
                bgcolor="rgba(255,255,255,0.75)",
            )

        # Nhan gia tri cuc tri trong long phan tu (khong phai 2 dau)
        idx_ext = int(np.argmax(np.abs(values)))
        if 0 < idx_ext < n - 1 and abs(values[idx_ext]) > 1e-6:
            fig.add_annotation(
                x=ox[idx_ext], y=oy[idx_ext], text=f"{values[idx_ext]:.3g}",
                showarrow=False, font=dict(size=10, color=diag_color),
                bgcolor="rgba(255,255,255,0.75)",
            )

    if hatch_x:
        fig.add_trace(go.Scatter(
            x=hatch_x, y=hatch_y, mode="lines",
            line=dict(color=diag_color, width=0.8),
            opacity=0.55, hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        template="plotly_white",
        height=460,
        margin=dict(l=50, r=30, t=60, b=50),
        xaxis=dict(zeroline=True, showgrid=True),
        yaxis=dict(zeroline=True, showgrid=True, scaleanchor="x", scaleratio=1),
        showlegend=False,
    )

    return fig

def plot_sfd_single(result: BeamResult) -> go.Figure:
    v_range = _padded_range(result.shear)
    fig = synced_figure("Shear Force Diagram", float(result.x[-1]), y_range=v_range, y_title="Shear (kN)")
    fig.add_trace(go.Scatter(x=result.x, y=result.shear, mode="lines",
                             fill="tozeroy", line={"color": COLOR_SFD, "width": 2},
                             fillcolor="rgba(11,95,255,0.20)",
                             hovertemplate="x=%{x:.2f}m  V=%{y:.2f}kN<extra></extra>"))
    fig.update_yaxes(fixedrange=True)
    return fig

def plot_bmd_single(result: BeamResult) -> go.Figure:
    m_range = _padded_range(result.moment)
    fig = synced_figure(
        "Bending Moment Diagram",
        float(result.x[-1]),
        y_range=m_range,
        y_title="Moment (kNm)"
    )
    fig.add_trace(go.Scatter(
        x=result.x,
        y=result.moment,
        mode="lines",
        fill="tozeroy",
        line={
            "color": COLOR_BMD,
            "width": 2
        },
        fillcolor="rgba(255,43,43,0.22)",
        hovertemplate=
        "x=%{x:.2f}m  M=%{y:.2f}kNm<extra></extra>"
    ))
    fig.update_yaxes(autorange="reversed", fixedrange=True)
    return fig
