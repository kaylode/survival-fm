# Data Analysis Report for mimiciv_survival_B.csv

## 1. Basic Information
- **Shape**: 58329 rows, 104 columns
- **Memory Usage**: 46.28 MB

## 2. Missing Values
Columns with missing data or -1.0 placeholders:

|                    |   NaN Count |   -1.0 Count |   Total % Missing |
|:-------------------|------------:|-------------:|------------------:|
| LAB_Alb_max        |       39820 |            0 |             68.27 |
| LAB_Alb_mean       |       39820 |            0 |             68.27 |
| LAB_Alb_min        |       39820 |            0 |             68.27 |
| LAB_PaO2_max       |       31209 |            0 |             53.51 |
| LAB_PaCO2_min      |       31210 |            0 |             53.51 |
| LAB_PaO2_min       |       31209 |            0 |             53.51 |
| LAB_PaO2_mean      |       31209 |            0 |             53.51 |
| LAB_PaCO2_mean     |       31210 |            0 |             53.51 |
| LAB_PaCO2_max      |       31210 |            0 |             53.51 |
| LAB_pH_mean        |       31021 |            0 |             53.18 |
| LAB_pH_min         |       31021 |            0 |             53.18 |
| PROC_dialysis      |       31020 |            0 |             53.18 |
| PROC_art_line      |       31020 |            0 |             53.18 |
| PROC_CVC           |       31020 |            0 |             53.18 |
| PROC_foley         |       31020 |            0 |             53.18 |
| PROC_NGT           |       31020 |            0 |             53.18 |
| PROC_CXR           |       31020 |            0 |             53.18 |
| PROC_niv           |       31020 |            0 |             53.18 |
| LAB_pH_max         |       31021 |            0 |             53.18 |
| PROC_invasive_vent |       31020 |            0 |             53.18 |
| PROC_ECG           |       31020 |            0 |             53.18 |
| LAB_Bili_max       |       30991 |            0 |             53.13 |
| LAB_Bili_mean      |       30991 |            0 |             53.13 |
| LAB_Bili_min       |       30991 |            0 |             53.13 |
| LAB_Lactate_mean   |       23950 |            0 |             41.06 |
| LAB_Lactate_min    |       23950 |            0 |             41.06 |
| LAB_Lactate_max    |       23950 |            0 |             41.06 |
| MED_insulin        |        9232 |            0 |             15.83 |
| MED_PRBC           |        9232 |            0 |             15.83 |
| MED_dopamine       |        9232 |            0 |             15.83 |
| MED_epinephrine    |        9232 |            0 |             15.83 |
| MED_heparin        |        9232 |            0 |             15.83 |
| MED_fentanyl       |        9232 |            0 |             15.83 |
| MED_norepinephrine |        9232 |            0 |             15.83 |
| MED_propofol       |        9232 |            0 |             15.83 |
| MED_midazolam      |        9232 |            0 |             15.83 |
| MED_pip_tazo       |        9232 |            0 |             15.83 |
| MED_furosemide     |        9232 |            0 |             15.83 |
| MED_normal_saline  |        9232 |            0 |             15.83 |
| MED_vancomycin     |        9232 |            0 |             15.83 |
| MED_vasopressin    |        9232 |            0 |             15.83 |
| MED_albumin_inf    |        9232 |            0 |             15.83 |
| VITAL_DBP_mean     |        2129 |            0 |              3.65 |
| VITAL_DBP_max      |        2129 |            0 |              3.65 |
| VITAL_DBP_min      |        2129 |            0 |              3.65 |
| VITAL_SBP_mean     |        2121 |            0 |              3.64 |
| VITAL_SBP_min      |        2121 |            0 |              3.64 |
| VITAL_SBP_max      |        2121 |            0 |              3.64 |
| urine_total_ml     |        1375 |            0 |              2.36 |
| LAB_WBC_mean       |        1239 |            0 |              2.12 |
| LAB_WBC_min        |        1239 |            0 |              2.12 |
| LAB_WBC_max        |        1239 |            0 |              2.12 |
| LAB_Plt_min        |        1225 |            0 |              2.1  |
| LAB_Plt_mean       |        1225 |            0 |              2.1  |
| LAB_Plt_max        |        1225 |            0 |              2.1  |
| LAB_Hgb_min        |        1219 |            0 |              2.09 |
| LAB_Hgb_max        |        1219 |            0 |              2.09 |
| LAB_Hgb_mean       |        1219 |            0 |              2.09 |
| LAB_Gluc_mean      |        1185 |            0 |              2.03 |
| LAB_Gluc_max       |        1185 |            0 |              2.03 |
| LAB_Gluc_min       |        1185 |            0 |              2.03 |
| LAB_BUN_min        |        1090 |            0 |              1.87 |
| LAB_BUN_max        |        1090 |            0 |              1.87 |
| LAB_BUN_mean       |        1090 |            0 |              1.87 |
| LAB_Cr_max         |        1077 |            0 |              1.85 |
| LAB_Cr_min         |        1077 |            0 |              1.85 |
| LAB_Cr_mean        |        1077 |            0 |              1.85 |
| LAB_Na_min         |        1075 |            0 |              1.84 |
| LAB_Na_mean        |        1075 |            0 |              1.84 |
| LAB_Na_max         |        1075 |            0 |              1.84 |
| LAB_HCO3_min       |        1073 |            0 |              1.84 |
| LAB_HCO3_mean      |        1073 |            0 |              1.84 |
| LAB_HCO3_max       |        1073 |            0 |              1.84 |
| VITAL_Temp_mean    |         774 |            0 |              1.33 |
| VITAL_Temp_min     |         774 |            0 |              1.33 |
| VITAL_Temp_max     |         774 |            0 |              1.33 |
| VITAL_MAP_min      |         166 |           51 |              0.37 |
| VITAL_MAP_max      |         166 |            0 |              0.28 |
| VITAL_MAP_mean     |         166 |            0 |              0.28 |
| VITAL_GCS_max      |         158 |            0 |              0.27 |
| VITAL_GCS_min      |         158 |            0 |              0.27 |
| VITAL_GCS_mean     |         158 |            0 |              0.27 |
| VITAL_RR_min       |         118 |            0 |              0.2  |
| VITAL_RR_mean      |         118 |            0 |              0.2  |
| VITAL_RR_max       |         118 |            0 |              0.2  |
| VITAL_SpO2_max     |          72 |            0 |              0.12 |
| VITAL_SpO2_mean    |          72 |            0 |              0.12 |
| VITAL_SpO2_min     |          72 |            0 |              0.12 |
| VITAL_HR_mean      |          55 |            0 |              0.09 |
| VITAL_HR_min       |          55 |            0 |              0.09 |
| VITAL_HR_max       |          55 |            0 |              0.09 |

## 3. Duplicates
- **Duplicate rows**: 0

## 4. Target and Survival Analysis
- **mortality distribution**:
  - 0: 91.73%
  - 1: 8.27%
- **survival_hours range**: [48.00666666666667, 7716.066666666667]
### Clinical Sanity Checks
- **Invalid ages (<0 or >120)**: 0
- **Patients with very short stay and mortality (<= 4h)**: 0

## 5. Numerical Column Issues
### Constant and Near-Constant Columns
#### Constant/Near-Constant (non-empty)
|    | Column        | Status   |   Value |   % Matching |
|---:|:--------------|:---------|--------:|-------------:|
|  0 | PROC_art_line | Constant |       0 |          100 |
|  1 | PROC_foley    | Constant |       0 |          100 |
|  2 | PROC_NGT      | Constant |       0 |          100 |
|  3 | PROC_ECG      | Constant |       0 |          100 |

### Potential Outliers (Z-score > 3)
|    | Column            |   Outliers |   % Outliers |
|---:|:------------------|-----------:|-------------:|
| 75 | race_black        |       5412 |         9.28 |
| 69 | MED_pip_tazo      |       4485 |         9.13 |
| 68 | MED_normal_saline |       4412 |         8.99 |
| 61 | VITAL_Temp_mean   |       3267 |         5.68 |
| 65 | MED_vasopressin   |       2100 |         4.28 |
| 66 | MED_epinephrine   |       2097 |         4.27 |
| 77 | race_hispanic     |       2057 |         3.53 |
| 70 | MED_albumin_inf   |       1876 |         3.82 |
| 76 | race_asian        |       1748 |         3    |
| 64 | VITAL_GCS_max     |       1556 |         2.67 |
|  5 | LAB_BUN_min       |       1365 |         2.38 |
| 11 | LAB_Cr_min        |       1245 |         2.17 |
| 63 | VITAL_GCS_mean    |       1058 |         1.82 |
| 67 | MED_dopamine      |       1009 |         2.06 |
| 71 | PROC_niv          |        984 |         3.6  |
| 26 | LAB_Na_min        |        880 |         1.54 |
| 17 | LAB_HCO3_min      |        741 |         1.29 |
| 59 | VITAL_SpO2_min    |        739 |         1.27 |
| 72 | PROC_dialysis     |        685 |         2.51 |
| 56 | VITAL_SBP_min     |        447 |         0.8  |
| 47 | VITAL_HR_min      |        399 |         0.68 |
| 73 | PROC_CXR          |        398 |         1.46 |
| 38 | LAB_WBC_min       |        370 |         0.65 |
| 62 | VITAL_Temp_min    |        328 |         0.57 |
| 20 | LAB_Hgb_min       |        141 |         0.25 |
| 46 | VITAL_HR_mean     |        126 |         0.22 |
| 53 | VITAL_RR_min      |        101 |         0.17 |
| 42 | VITAL_DBP_max     |         76 |         0.14 |
| 74 | urine_total_ml    |         50 |         0.09 |
| 60 | VITAL_Temp_max    |         33 |         0.06 |
|  3 | LAB_BUN_max       |         32 |         0.06 |
| 12 | LAB_Gluc_max      |         32 |         0.06 |
| 13 | LAB_Gluc_mean     |         32 |         0.06 |
|  4 | LAB_BUN_mean      |         32 |         0.06 |
| 19 | LAB_Hgb_mean      |         28 |         0.05 |
| 18 | LAB_Hgb_max       |         28 |         0.05 |
| 40 | LAB_pH_mean       |         27 |         0.1  |
| 39 | LAB_pH_max        |         27 |         0.1  |
| 36 | LAB_WBC_max       |         26 |         0.05 |
| 37 | LAB_WBC_mean      |         26 |         0.05 |
| 43 | VITAL_DBP_mean    |         25 |         0.04 |
| 30 | LAB_PaO2_max      |         24 |         0.09 |
| 31 | LAB_PaO2_mean     |         24 |         0.09 |
| 28 | LAB_PaCO2_mean    |         24 |         0.09 |
| 27 | LAB_PaCO2_max     |         24 |         0.09 |
| 33 | LAB_Plt_max       |         24 |         0.04 |
| 34 | LAB_Plt_mean      |         24 |         0.04 |
| 21 | LAB_Lactate_max   |         23 |         0.07 |
| 22 | LAB_Lactate_mean  |         23 |         0.07 |
| 15 | LAB_HCO3_max      |         15 |         0.03 |
| 16 | LAB_HCO3_mean     |         15 |         0.03 |
| 10 | LAB_Cr_mean       |         13 |         0.02 |
|  9 | LAB_Cr_max        |         13 |         0.02 |
| 24 | LAB_Na_max        |         11 |         0.02 |
| 25 | LAB_Na_mean       |         11 |         0.02 |
| 55 | VITAL_SBP_mean    |          9 |         0.02 |
| 48 | VITAL_MAP_max     |          7 |         0.01 |
| 57 | VITAL_SpO2_max    |          6 |         0.01 |
| 58 | VITAL_SpO2_mean   |          6 |         0.01 |
| 45 | VITAL_HR_max      |          6 |         0.01 |
| 54 | VITAL_SBP_max     |          5 |         0.01 |
| 52 | VITAL_RR_mean     |          4 |         0.01 |
| 51 | VITAL_RR_max      |          4 |         0.01 |
| 49 | VITAL_MAP_mean    |          4 |         0.01 |
| 32 | LAB_PaO2_min      |          3 |         0.01 |
| 29 | LAB_PaCO2_min     |          3 |         0.01 |
|  6 | LAB_Bili_max      |          3 |         0.01 |
|  7 | LAB_Bili_mean     |          3 |         0.01 |
|  1 | LAB_Alb_mean      |          2 |         0.01 |
|  0 | LAB_Alb_max       |          2 |         0.01 |
| 35 | LAB_Plt_min       |          2 |         0    |
| 41 | LAB_pH_min        |          2 |         0.01 |
| 23 | LAB_Lactate_min   |          2 |         0.01 |
|  2 | LAB_Alb_min       |          1 |         0.01 |
|  8 | LAB_Bili_min      |          1 |         0    |
| 14 | LAB_Gluc_min      |          1 |         0    |
| 44 | VITAL_DBP_min     |          1 |         0    |
| 50 | VITAL_MAP_min     |          1 |         0    |

## 6. Correlation Analysis
### Highly Collinear Features (> 0.9)
|    | Var1            | Var2            |   Correlation |
|---:|:----------------|:----------------|--------------:|
|  0 | LAB_Alb_mean    | LAB_Alb_max     |        0.9487 |
|  1 | LAB_BUN_mean    | LAB_BUN_max     |        0.934  |
|  2 | LAB_Bili_max    | LAB_Alb_max     |        1      |
|  3 | LAB_Bili_max    | LAB_Alb_mean    |        0.9487 |
|  4 | LAB_Bili_mean   | LAB_Alb_mean    |        0.9762 |
|  5 | LAB_Bili_mean   | LAB_Alb_min     |        0.9701 |
|  6 | LAB_Bili_min    | LAB_Alb_min     |        1      |
|  7 | LAB_Bili_min    | LAB_Bili_mean   |        0.9231 |
|  8 | LAB_Cr_mean     | LAB_Cr_max      |        0.9486 |
|  9 | LAB_HCO3_mean   | LAB_HCO3_max    |        0.9111 |
| 10 | LAB_Na_mean     | LAB_Na_max      |        0.9078 |
| 11 | LAB_PaO2_max    | LAB_PaCO2_max   |        1      |
| 12 | LAB_PaO2_mean   | LAB_PaCO2_mean  |        1      |
| 13 | LAB_PaO2_min    | LAB_PaCO2_min   |        1      |
| 14 | LAB_WBC_mean    | LAB_WBC_max     |        0.9208 |
| 15 | LAB_pH_max      | LAB_PaCO2_max   |        0.9034 |
| 16 | LAB_pH_max      | LAB_PaO2_max    |        0.9035 |
| 17 | LAB_pH_mean     | LAB_PaCO2_mean  |        0.9323 |
| 18 | LAB_pH_mean     | LAB_PaO2_mean   |        0.9323 |
| 19 | LAB_pH_min      | LAB_Lactate_min |        1      |
| 20 | VITAL_DBP_min   | VITAL_DBP_mean  |        0.9463 |
| 21 | VITAL_MAP_mean  | VITAL_MAP_max   |        0.9948 |
| 22 | VITAL_RR_mean   | VITAL_RR_max    |        0.9977 |
| 23 | VITAL_SBP_mean  | VITAL_SBP_max   |        0.9634 |
| 24 | VITAL_SpO2_mean | VITAL_SpO2_max  |        0.9855 |
| 25 | VITAL_Temp_min  | VITAL_Temp_mean |        0.91   |

### [SUSPICIOUS] Extremely High Correlation (> 0.999, non-ID)
These might be redundant or derived features.

|    | Var1          | Var2            |   Correlation |
|---:|:--------------|:----------------|--------------:|
|  0 | LAB_Bili_max  | LAB_Alb_max     |      1        |
|  1 | LAB_Bili_min  | LAB_Alb_min     |      1        |
|  2 | LAB_PaO2_max  | LAB_PaCO2_max   |      0.999991 |
|  3 | LAB_PaO2_mean | LAB_PaCO2_mean  |      0.999981 |
|  4 | LAB_PaO2_min  | LAB_PaCO2_min   |      0.999987 |
|  5 | LAB_pH_min    | LAB_Lactate_min |      1        |

### Top Correlations with mortality
|                    |   mortality |
|:-------------------|------------:|
| mortality          |   1         |
| MED_norepinephrine |   0.200986  |
| MED_vasopressin    |   0.182741  |
| LAB_BUN_min        |   0.181486  |
| MED_fentanyl       |   0.176     |
| MED_midazolam      |   0.126547  |
| MED_vancomycin     |   0.108608  |
| MED_albumin_inf    |   0.0987284 |
| age                |   0.0980742 |
| VITAL_HR_mean      |   0.0927832 |

|                |   mortality |
|:---------------|------------:|
| LAB_Hgb_min    |  -0.0906561 |
| VITAL_SpO2_min |  -0.108148  |
| VITAL_SBP_min  |  -0.116731  |
| VITAL_GCS_min  |  -0.14364   |
| VITAL_GCS_mean |  -0.29762   |
| VITAL_GCS_max  |  -0.339084  |
| PROC_art_line  | nan         |
| PROC_foley     | nan         |
| PROC_NGT       | nan         |
| PROC_ECG       | nan         |

## 7. Categorical Column Analysis
No categorical columns found.

## 8. Bin Count (Unique Values / Discretization)
Unique values count for numeric columns (high unique = continuous, low unique = ordinal/binned):

|    | Column             |   Unique Values |
|---:|:-------------------|----------------:|
| 87 | PROC_foley         |               1 |
| 85 | PROC_art_line      |               1 |
| 90 | PROC_ECG           |               1 |
| 88 | PROC_NGT           |               1 |
|  1 | mortality          |               2 |
| 73 | MED_propofol       |               2 |
| 74 | MED_fentanyl       |               2 |
| 75 | MED_heparin        |               2 |
| 81 | MED_albumin_inf    |               2 |
| 80 | MED_pip_tazo       |               2 |
| 82 | PROC_invasive_vent |               2 |
| 94 | race_black         |               2 |
| 96 | race_hispanic      |               2 |
| 97 | ins_medicare       |               2 |
| 93 | race_white         |               2 |
| 95 | race_asian         |               2 |
| 72 | MED_dopamine       |               2 |
| 69 | MED_norepinephrine |               2 |
| 71 | MED_epinephrine    |               2 |
| 70 | MED_vasopressin    |               2 |
| 83 | PROC_niv           |               2 |
| 79 | MED_vancomycin     |               2 |
| 77 | MED_normal_saline  |               2 |
| 78 | MED_PRBC           |               2 |
| 76 | MED_insulin        |               2 |
| 89 | PROC_CXR           |               2 |
| 84 | PROC_dialysis      |               2 |
| 86 | PROC_CVC           |               2 |
| 92 | gender_male        |               2 |
| 67 | VITAL_GCS_min      |              13 |
| 68 | VITAL_GCS_max      |              13 |
| 56 | VITAL_RR_min       |              33 |
|  3 | LAB_Alb_max        |              50 |
|  5 | LAB_Alb_min        |              55 |
| 60 | VITAL_SpO2_max     |              60 |
| 27 | LAB_Na_max         |              74 |
| 18 | LAB_HCO3_max       |              76 |
| 42 | LAB_pH_max         |              77 |
| 29 | LAB_Na_min         |              81 |
| 20 | LAB_HCO3_min       |              83 |
|  2 | age                |              86 |
| 44 | LAB_pH_min         |              87 |
| 62 | VITAL_SpO2_min     |              97 |
| 32 | LAB_PaCO2_min      |              98 |
| 47 | VITAL_DBP_min      |             115 |
| 54 | VITAL_RR_max       |             129 |
| 26 | LAB_Lactate_min    |             131 |
| 50 | VITAL_HR_min       |             134 |
| 30 | LAB_PaCO2_max      |             138 |
| 21 | LAB_Hgb_max        |             162 |
| 59 | VITAL_SBP_min      |             169 |
| 53 | VITAL_MAP_min      |             169 |
| 23 | LAB_Hgb_min        |             169 |
| 14 | LAB_Cr_min         |             173 |
|  8 | LAB_BUN_min        |             178 |
| 48 | VITAL_HR_max       |             201 |
| 12 | LAB_Cr_max         |             205 |
| 24 | LAB_Lactate_max    |             208 |
|  6 | LAB_BUN_max        |             212 |
| 57 | VITAL_SBP_max      |             222 |
| 63 | VITAL_Temp_max     |             274 |
| 65 | VITAL_Temp_min     |             300 |
| 45 | VITAL_DBP_max      |             308 |
| 11 | LAB_Bili_min       |             364 |
| 17 | LAB_Gluc_min       |             370 |
|  9 | LAB_Bili_max       |             394 |
| 51 | VITAL_MAP_max      |             395 |
|  4 | LAB_Alb_mean       |             405 |
| 35 | LAB_PaO2_min       |             458 |
| 41 | LAB_WBC_min        |             556 |
| 33 | LAB_PaO2_max       |             607 |
| 38 | LAB_Plt_min        |             733 |
| 39 | LAB_WBC_max        |             742 |
| 15 | LAB_Gluc_max       |             774 |
| 36 | LAB_Plt_max        |             828 |
| 28 | LAB_Na_mean        |            1365 |
| 19 | LAB_HCO3_mean      |            1424 |
| 10 | LAB_Bili_mean      |            1700 |
| 13 | LAB_Cr_mean        |            2303 |
| 31 | LAB_PaCO2_mean     |            2529 |
| 43 | LAB_pH_mean        |            2561 |
|  7 | LAB_BUN_mean       |            2580 |
| 25 | LAB_Lactate_mean   |            2906 |
| 22 | LAB_Hgb_mean       |            3603 |
| 16 | LAB_Gluc_mean      |            4367 |
| 40 | LAB_WBC_mean       |            5702 |
| 66 | VITAL_GCS_mean     |            6062 |
| 91 | urine_total_ml     |            6341 |
| 37 | LAB_Plt_mean       |            6553 |
| 34 | LAB_PaO2_mean      |            7728 |
| 61 | VITAL_SpO2_mean    |           10367 |
| 64 | VITAL_Temp_mean    |           11193 |
| 55 | VITAL_RR_mean      |           16394 |
| 46 | VITAL_DBP_mean     |           22197 |
| 58 | VITAL_SBP_mean     |           25269 |
| 52 | VITAL_MAP_mean     |           29867 |
| 49 | VITAL_HR_mean      |           32991 |
|  0 | survival_hours     |           49678 |

