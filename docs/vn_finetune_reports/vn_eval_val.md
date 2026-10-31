# So sánh model trên dữ liệu VN (val, mức cửa sổ 1 s)

Cửa sổ VN: 1453, cửa sổ gốc (ESC-50/US8K/FSD50K): 4530.

## Dữ liệu VN

| Chỉ số | cu | moi |
|---|---|---|
| macro-F1 | 0.661 | 0.686 |
| recall `smoke_alarm` (114) | 0.561 | 0.842 |
| recall `car_horn` (20) | 0.100 | 0.100 |
| recall `siren` (564) | 0.860 | 0.872 |
| recall `background` (755) | 0.905 | 0.921 |
| nền → báo động (tỷ lệ cửa sổ) | 0.095 | 0.079 |
| nền → siren | 0.011 | 0.015 |

## Dữ liệu gốc

| Chỉ số | cu | moi |
|---|---|---|
| macro-F1 | 0.805 | 0.787 |
| recall `smoke_alarm` (67) | 0.866 | 0.940 |
| recall `car_horn` (128) | 0.773 | 0.727 |
| recall `siren` (625) | 0.962 | 0.965 |
| recall `baby_cry` (101) | 0.851 | 0.842 |
| recall `door_knock` (35) | 0.771 | 0.829 |
| recall `dog_bark` (568) | 0.754 | 0.762 |
| recall `water_running` (194) | 0.727 | 0.665 |
| recall `background` (2812) | 0.896 | 0.887 |
| nền → báo động (tỷ lệ cửa sổ) | 0.104 | 0.113 |
| nền → siren | 0.017 | 0.013 |

## Recall còi VN theo nhóm

| Nhóm | cu | moi |
|---|---|---|
| vehicle_type=ambulance (144 cửa sổ) | 0.882 | 0.833 |
| vehicle_type=fire (240 cửa sổ) | 0.867 | 0.908 |
| vehicle_type=other (36 cửa sổ) | 0.917 | 0.917 |
| vehicle_type=police (144 cửa sổ) | 0.812 | 0.840 |
| passing=False (510 cửa sổ) | 0.873 | 0.886 |
| passing=True (54 cửa sổ) | 0.741 | 0.741 |
| distance=far (228 cửa sổ) | 0.825 | 0.860 |
| distance=mid (174 cửa sổ) | 0.879 | 0.925 |
| distance=near (162 cửa sổ) | 0.889 | 0.833 |

