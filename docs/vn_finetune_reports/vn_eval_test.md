# So sánh model trên dữ liệu VN (test, mức cửa sổ 1 s)

Cửa sổ VN: 1359, cửa sổ gốc (ESC-50/US8K/FSD50K): 4631.

## Dữ liệu VN

| Chỉ số | cu | moi |
|---|---|---|
| macro-F1 | 0.670 | 0.690 |
| recall `smoke_alarm` (137) | 0.664 | 0.686 |
| recall `car_horn` (20) | 0.150 | 0.150 |
| recall `siren` (522) | 0.828 | 0.931 |
| recall `background` (680) | 0.876 | 0.882 |
| nền → báo động (tỷ lệ cửa sổ) | 0.124 | 0.118 |
| nền → siren | 0.006 | 0.006 |

## Dữ liệu gốc

| Chỉ số | cu | moi |
|---|---|---|
| macro-F1 | 0.784 | 0.793 |
| recall `smoke_alarm` (90) | 0.900 | 0.856 |
| recall `car_horn` (191) | 0.733 | 0.754 |
| recall `siren` (617) | 0.561 | 0.580 |
| recall `baby_cry` (195) | 0.815 | 0.774 |
| recall `door_knock` (42) | 0.857 | 0.881 |
| recall `dog_bark` (571) | 0.779 | 0.795 |
| recall `water_running` (188) | 0.713 | 0.755 |
| recall `background` (2737) | 0.878 | 0.885 |
| nền → báo động (tỷ lệ cửa sổ) | 0.122 | 0.115 |
| nền → siren | 0.037 | 0.029 |

## Recall còi VN theo nhóm

| Nhóm | cu | moi |
|---|---|---|
| vehicle_type=ambulance (144 cửa sổ) | 0.972 | 0.938 |
| vehicle_type=fire (234 cửa sổ) | 0.688 | 0.902 |
| vehicle_type=other (36 cửa sổ) | 0.972 | 0.972 |
| vehicle_type=police (108 cửa sổ) | 0.889 | 0.972 |
| passing=False (498 cửa sổ) | 0.831 | 0.932 |
| passing=True (24 cửa sổ) | 0.750 | 0.917 |
| distance=far (156 cửa sổ) | 0.904 | 0.942 |
| distance=mid (168 cửa sổ) | 0.792 | 0.970 |
| distance=near (198 cửa sổ) | 0.798 | 0.889 |

