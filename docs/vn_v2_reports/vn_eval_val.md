# So sánh model trên dữ liệu VN (val, mức cửa sổ 1 s)

Cửa sổ VN: 1453, cửa sổ gốc (ESC-50/US8K/FSD50K): 4530.

## Dữ liệu VN

| Chỉ số | cu | vn1 | vn2 |
|---|---|---|---|
| macro-F1 | 0.661 | 0.686 | 0.707 |
| recall `smoke_alarm` (114) | 0.561 | 0.842 | 0.789 |
| recall `car_horn` (20) | 0.100 | 0.100 | 0.100 |
| recall `siren` (564) | 0.860 | 0.872 | 0.936 |
| recall `background` (755) | 0.905 | 0.921 | 0.911 |
| nền → báo động (tỷ lệ cửa sổ) | 0.095 | 0.079 | 0.089 |
| nền → siren | 0.011 | 0.015 | 0.012 |
| ECE (top-label) | 0.086 | 0.049 | 0.026 |
| ECE (từng lớp, TB) | 0.032 | 0.021 | 0.022 |
| tỷ lệ cửa sổ bão hoà ≥255/256 | 0.477 | 0.400 | 0.217 |

## Dữ liệu gốc

| Chỉ số | cu | vn1 | vn2 |
|---|---|---|---|
| macro-F1 | 0.805 | 0.787 | 0.798 |
| recall `smoke_alarm` (67) | 0.866 | 0.940 | 0.970 |
| recall `car_horn` (128) | 0.773 | 0.727 | 0.789 |
| recall `siren` (625) | 0.962 | 0.965 | 0.968 |
| recall `baby_cry` (101) | 0.851 | 0.842 | 0.881 |
| recall `door_knock` (35) | 0.771 | 0.829 | 0.743 |
| recall `dog_bark` (568) | 0.754 | 0.762 | 0.762 |
| recall `water_running` (194) | 0.727 | 0.665 | 0.701 |
| recall `background` (2812) | 0.896 | 0.887 | 0.892 |
| nền → báo động (tỷ lệ cửa sổ) | 0.104 | 0.113 | 0.108 |
| nền → siren | 0.017 | 0.013 | 0.021 |
| ECE (top-label) | 0.064 | 0.067 | 0.032 |
| ECE (từng lớp, TB) | 0.020 | 0.022 | 0.021 |
| tỷ lệ cửa sổ bão hoà ≥255/256 | 0.576 | 0.569 | 0.294 |

## Recall còi VN theo nhóm

| Nhóm | cu | vn1 | vn2 |
|---|---|---|---|
| vehicle_type=ambulance (144 cửa sổ) | 0.882 | 0.833 | 0.958 |
| vehicle_type=fire (240 cửa sổ) | 0.867 | 0.908 | 0.942 |
| vehicle_type=other (36 cửa sổ) | 0.917 | 0.917 | 0.917 |
| vehicle_type=police (144 cửa sổ) | 0.812 | 0.840 | 0.910 |
| passing=False (510 cửa sổ) | 0.873 | 0.886 | 0.943 |
| passing=True (54 cửa sổ) | 0.741 | 0.741 | 0.870 |
| distance=far (228 cửa sổ) | 0.825 | 0.860 | 0.921 |
| distance=mid (174 cửa sổ) | 0.879 | 0.925 | 0.954 |
| distance=near (162 cửa sổ) | 0.889 | 0.833 | 0.938 |

