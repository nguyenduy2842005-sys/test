# Đớp Tool — tối ưu hoá kết cấu thép bằng Phép tính Biến phân

Chạy: `pip install -r requirements.txt` rồi `streamlit run web_app.py`. Kiểm thử: `python -m pytest -q tests`.

## Cấu trúc
| File | Vai trò |
|---|---|
| `fem_core.py` | Core FEM của bạn, **giữ nguyên** (ma trận phần tử, ma trận xoay, tải tương đương, nội lực). |
| `opt_core.py` | Lõi tối ưu: biến phân Euler–Lagrange + FEM-OC/KKT. |
| `web_app.py` | Giao diện Streamlit: Lý thuyết · Cột · Dầm vút thon · Toàn khung/dàn. |
| `canvas/index.html` | Canvas vẽ kết cấu (nút/thanh/gối). |
| `tests/test_core.py` | Kiểm chứng số học. |

## Trực quan hoá trước / sau tối ưu
Cột, dầm và từng thanh khung/dàn đều có hình **trước (nét đứt)** so với **sau (nét liền, tô màu)**: xanh lá = vật liệu bỏ được, cam = chỗ dày hơn mốc.
Mốc cột: cùng P_cr hoặc cùng thể tích (chọn được). Mốc dầm và khung/dàn: tiết diện đều nhỏ nhất thoả cùng ràng buộc.
Kích thước vẽ ∝ A^γ (γ của họ tiết diện; γ = 0 như chữ I cao cố định thì vẽ ∝ A). Với cùng thể tích, nghiệm tối ưu của cột luôn phình ra ngoài mốc ở giữa nhịp.

## Phương pháp
1. **Cấu kiện biến thiên A(x), I = α·Aⁿ.**
   - *Cột:* Euler–Lagrange cho `A ∝ |w|^(2/(n+1))`, nghiệm đóng qua hàm Beta không đầy đủ; cột khớp–khớp n=2 cho tỷ số tải tới hạn 4/3 (Keller).
   - *Dầm tĩnh định:* F không chứa A′ nên Euler–Lagrange suy biến thành phương trình đại số `A = [λ·n·M·m̄/(Eα)]^(1/(n+1))`; λ giải từ ràng buộc võng.
   - *Dầm siêu tĩnh:* M phụ thuộc A nên không có nghiệm đóng, dùng FEM + KKT rời rạc.
2. **Toàn khung/dàn:** FEM (tái dùng `fem_core`) + ràng buộc bền/ổn định Euler (FSD) + ràng buộc chuyển vị (OC, rồi SLSQP với đạo hàm liên hợp).

## Kiểm chứng đã làm
- Cột đều: FEM khớp công thức Euler (sai số < 1e-4) cho 3 sơ đồ.
- Cột tối ưu khớp–khớp: tỷ số 4/3 chính xác; FEM trên profile rời rạc lệch < 1% so với nghiệm đóng.
- Dầm tựa đơn: thể tích FEM-OC lệch < 2% so với nghiệm đóng E–L.
- Bộ giải nhanh trùng `fem_core.solve_plane_frame` (sai khác tương đối ≤ 1e-10).
- Phản lực cân bằng ΣFx, ΣFy, ΣM ≈ 1e-11.
- **Dàn 10 thanh chuẩn (Schmit):** 5076.7 lb so với cực trị đã biết 5060.85 lb (+0.3%).

## Giới hạn đã biết
- Dàn/khung **siêu tĩnh** có thể dừng ở điểm dừng cục bộ cách cực trị vài phần nghìn (cận bền A_lb được cố định theo nội lực mỗi vòng, không tính độ nhạy nội lực theo A).
- Euler–Bernoulli, đàn hồi; chưa xét P-Δ, ổn định cục bộ, mỏi, tải bản thân, liên kết. Tiết diện thực cần làm tròn theo thép hình rồi kiểm tra lại theo TCVN/AISC.

## Lưu ý về `fem_core.py`
`solve_plane_frame` báo **phản lực sai dấu** (lực xuống 10 kN ở đầu console cho Ry = −10, và phần "Kiểm tra ΣFy" ra −20 thay vì 0). Core được giữ nguyên; lõi tối ưu dùng phản lực riêng đã đúng dấu.

## Biểu đồ tải trọng · lực cắt · mô men · chuyển vị (ghép từ web cũ)
- `legacy_plots.py`: trích nguyên văn các hàm vẽ từ `web_app_v2.py` (sơ đồ tải, SFD, BMD, đường đàn hồi, biểu đồ khung).
- `viz_results.py`: cầu nối giữa kết quả tối ưu và các hàm vẽ cũ; nét "trước tối ưu" là nét đứt **trắng** trên nền tối (đen trên nền sáng).
- Tab Dầm: lưới 2×2 (tải / V / M / võng) + phản lực. Tab Khung/dàn: các tab Sơ đồ tải, Nội lực (M/V/N), Chuyển vị.
- Quy ước: tải hướng xuống > 0; V dương khi phần bên trái đẩy lên; M > 0 căng thớ dưới (vẽ xuống dưới); võng xuống > 0.
- Lưu ý: `fem_core.solve_plane_frame` trả dấu phản lực ngược (giữ nguyên core); `opt_core.reactions()` đã hiệu chỉnh. Cột chưa có biểu đồ V/M.
