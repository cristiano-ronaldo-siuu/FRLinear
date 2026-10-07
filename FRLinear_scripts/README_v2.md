# FRLinear v2: hướng dẫn chạy

## 0. Chuẩn bị (trên máy thuê)

```bash
git clone -b FRLinear https://github.com/phamchivy/LTSF-Linear-vypc1.git && cd LTSF-Linear-vypc1
# chép các file trong gói FRLinear_v2 đè lên repo, xóa các file trong DELETED.txt
pip install torch pandas scikit-learn scipy statsmodels matplotlib
```

Đặt 9 file vào `./dataset/` và giữ **đúng tên**:
`ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv national_illness.csv exchange_rate.csv weather.csv electricity.csv traffic.csv`

Nên chạy trong `tmux` để không mất tiến trình khi mất kết nối SSH: `tmux new -s fr` (thoát ra: Ctrl-b d, vào lại: `tmux attach -t fr`).

## 1. Chạy thử nhanh (khoảng 5 phút, bắt buộc trước khi chạy đầy đủ)

```bash
bash FRLinear_scripts/run_all.sh --smoke --datasets ETTh1,ILI,Electricity
```
Chế độ smoke chạy 1 epoch, 1 run và chỉ horizon đầu tiên. Kết quả ghi vào `results_json_smoke/` và `results_summary_smoke/`.

Cần kiểm tra:
- Không có dòng `FAIL`.
- `results_summary_smoke/sanity_check.csv`: cột `rel_diff_pct` nằm trong khoảng ±1%. Chạy FRLinear với k=1, MA-25, không norm phải ra gần đúng DLinear.
- Dòng ETA in ra, dùng để ước lượng thời gian chạy đầy đủ.

## 2. Chạy đầy đủ

```bash
nohup bash FRLinear_scripts/run_all.sh > run_all.out 2>&1 &
tail -f logs/run_all_grid_*.log        # tiến độ + ETA
```

Pipeline gồm 5 bước:
1. Sinh regime (chỉ dùng tập train).
2. Grid: DLinear, NLinear, FRLinear k∈{1,2,3} × norm∈{none,last}.
3. Chọn (k, norm) theo val.
4. Ablation.
5. Xuất bảng.

**Có thể dừng (Ctrl-C hoặc kill) rồi chạy lại cùng lệnh bất kỳ lúc nào.** Các run đã xong (có file json) sẽ được bỏ qua.

Thứ tự chạy là ETTh1 → ETTh2 → ETTm1 → ETTm2 → ILI → Exchange → Weather → Electricity → Traffic. Nếu hết giờ, phần thiếu sẽ là Traffic.

Các tùy chọn hữu ích (thêm vào sau `run_all.sh`):

| Tùy chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--datasets ETTh1,ETTm1` | all | chỉ chạy một số dataset |
| `--horizons 96,192` | tất cả | chỉ chạy một số horizon |
| `--max_jobs_small / _medium / _big` | 6 / 4 / 2 | số tiến trình song song |
| `--runs` | 5 | số run cho baseline và cấu hình FRLinear được chọn |
| `--grid_runs` | 3 | số run cho mỗi cấu hình trong grid chọn (k, norm) |
| `--runs_big`, `--grid_runs_big` | 3, 2 | tương tự, cho Electricity và Traffic |

Nếu đổi `--grid_runs` hoặc `--grid_runs_big`, phải truyền cùng giá trị cho `select_config.py`.

**Số job với mặc định:** khoảng 160 cho mỗi dataset nhỏ (ETT, ILI, Exchange, Weather) và khoảng 92 cho Electricity và Traffic, tổng cộng khoảng 1300 job.

## 3. Kết quả (`results_summary/`)

| File | Dùng cho |
|---|---|
| `main_table.tex` | Bảng chính: NLinear, DLinear, FRLinear (cấu hình đã chọn), mean±std, tự in đậm và gạch chân |
| `ablation_mse.tex`, `ablation_mae.tex` | Ablation: k=1/2/3, random grouping, Fixed MA-25, đảo norm |
| `selection_table.tex` | Separability so với k được chọn và mức gain (trả lời R4) |
| `selected.json`, `selection_report.json` | Cấu hình được chọn và val loss của mọi cấu hình |
| `kernel_weights.csv` | Trọng số kernel đã học theo nhóm (phân tích C2) |
| `summary.csv`, `all_runs.csv` | Toàn bộ số liệu thô |
| `sanity_check.csv` | FRLinear k=1, MA-25 so với DLinear |

Vẽ hình: `python FRLinear/data-analysis/src/dataset_analysis/06_visualize_regime_separability.py`, kết quả nằm trong `output_figures/`.

Log từng run: `logs/YYYY-MM-DD/<dataset>/<model_id>_run<r>.log`. Kết quả từng run: `results_json/*.json`, gồm thời điểm chạy, seed, val loss, số epoch, thời gian train và các group.

## 4. Chạy một cấu hình bằng tay

```bash
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --data ETTh1 \
  --features M --seq_len 336 --pred_len 96 --enc_in 7 --batch_size 32 --learning_rate 0.005 \
  --train_epochs 50 --itr 1 --model FRLinear --model_id test --num_workers 0 \
  --n_regimes 2 --norm last --regime_mode ward --decomp adaptive --debug
```

Các cờ của FRLinear:

| Cờ | Giá trị |
|---|---|
| `--n_regimes` | 1, 2, 3 |
| `--regime_mode` | `ward`, `random` (kèm `--rand_seed`) |
| `--decomp` | `adaptive`, `fixed` |
| `--kernel_bank` | `5,9,13,21,31,41` |
| `--pad` | `replicate`, `zero` |
| `--norm` | `none`, `last` |
| `--gate_delta` | 0.005 |
| `--seed` | 2021 (run r dùng seed + r) |

## 5. Những điểm phải cập nhật trong paper

- **Regime:** đặc trưng ACF/FFT chỉ tính trên tập train. FFT bỏ thành phần trung bình (de-mean) trước khi tính. Chu kỳ mùa vụ và số lag theo từng dataset (lấy từ registry `FRLinear/datasets.py`).
- **Phân cụm:** Ward agglomerative trên (trend, season) đã chuẩn hóa, cho phân hoạch lồng nhau với k = 1–3. Bỏ ràng buộc m_min, cho phép nhóm 1 kênh.
- **Kernel bank:** dùng chung {5, 9, 13, 21, 31, 41} cho mọi nhóm. Moving average đệm kiểu replicate (giống DLinear), không còn zero padding.
- **Norm:** tùy chọn trừ giá trị cuối kiểu NLinear.
- **Chọn cấu hình:** (k, norm) chọn theo val loss trung bình qua các horizon và các run của grid, dùng quy tắc one-standard-error (hòa thì chọn k nhỏ hơn, rồi chọn không norm).
- **Thiết lập thí nghiệm:**
  - Mọi model chạy trong cùng một code base, test loader để `drop_last=False`.
  - lr và batch size theo script chính thức của DLinear cho từng dataset và horizon, áp dụng giống nhau cho cả 3 model.
  - 50 epoch, patience 3.
  - Seed 2021 + r.
