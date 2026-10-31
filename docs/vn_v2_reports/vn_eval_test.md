# So sánh model trên dữ liệu VN (test, mức cửa sổ 1 s)

Cửa sổ VN: 1359, cửa sổ gốc (ESC-50/US8K/FSD50K): 4631.

## Dữ liệu VN

| Chỉ số | cu | vn1 | vn2 |
|---|---|---|---|
| macro-F1 | 0.670 | 0.690 | 0.707 |
| recall `smoke_alarm` (137) | 0.664 | 0.686 | 0.745 |
| recall `car_horn` (20) | 0.150 | 0.150 | 0.150 |
| recall `siren` (522) | 0.828 | 0.931 | 0.983 |
| recall `background` (680) | 0.876 | 0.882 | 0.862 |
| nền → báo động (tỷ lệ cửa sổ) | 0.124 | 0.118 | 0.138 |
| nền → siren | 0.006 | 0.006 | 0.015 |
| ECE (top-label) | 0.109 | 0.065 | 0.022 |
| ECE (từng lớp, TB) | 0.036 | 0.026 | 0.028 |
| tỷ lệ cửa sổ bão hoà ≥255/256 | 0.486 | 0.443 | 0.233 |

## Dữ liệu gốc

| Chỉ số | cu | vn1 | vn2 |
|---|---|---|---|
| macro-F1 | 0.784 | 0.793 | 0.768 |
| recall `smoke_alarm` (90) | 0.900 | 0.856 | 0.844 |
| recall `car_horn` (191) | 0.733 | 0.754 | 0.775 |
| recall `siren` (617) | 0.561 | 0.580 | 0.580 |
| recall `baby_cry` (195) | 0.815 | 0.774 | 0.821 |
| recall `door_knock` (42) | 0.857 | 0.881 | 0.762 |
| recall `dog_bark` (571) | 0.779 | 0.795 | 0.760 |
| recall `water_running` (188) | 0.713 | 0.755 | 0.729 |
| recall `background` (2737) | 0.878 | 0.885 | 0.867 |
| nền → báo động (tỷ lệ cửa sổ) | 0.122 | 0.115 | 0.133 |
| nền → siren | 0.037 | 0.029 | 0.042 |
| ECE (top-label) | 0.111 | 0.100 | 0.046 |
| ECE (từng lớp, TB) | 0.033 | 0.032 | 0.026 |
| tỷ lệ cửa sổ bão hoà ≥255/256 | 0.498 | 0.483 | 0.231 |

## Recall còi VN theo nhóm

| Nhóm | cu | vn1 | vn2 |
|---|---|---|---|
| vehicle_type=ambulance (144 cửa sổ) | 0.972 | 0.938 | 0.986 |
| vehicle_type=fire (234 cửa sổ) | 0.688 | 0.902 | 0.983 |
| vehicle_type=other (36 cửa sổ) | 0.972 | 0.972 | 0.972 |
| vehicle_type=police (108 cửa sổ) | 0.889 | 0.972 | 0.981 |
| passing=False (498 cửa sổ) | 0.831 | 0.932 | 0.984 |
| passing=True (24 cửa sổ) | 0.750 | 0.917 | 0.958 |
| distance=far (156 cửa sổ) | 0.904 | 0.942 | 0.974 |
| distance=mid (168 cửa sổ) | 0.792 | 0.970 | 0.988 |
| distance=near (198 cửa sổ) | 0.798 | 0.889 | 0.985 |

