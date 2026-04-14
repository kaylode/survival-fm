# Data Analysis Report for eicu_survival.csv

## 1. Basic Information
- **Shape**: 66719 rows, 134 columns
- **Memory Usage**: 71.39 MB

## 2. Missing Values
Columns with missing data or -1.0 placeholders:

|                         |   NaN Count |   -1.0 Count |   Total % Missing |
|:------------------------|------------:|-------------:|------------------:|
| ICD10_chap_XVI          |       58201 |            0 |             87.23 |
| ICD10_chap_XV           |       58201 |            0 |             87.23 |
| ICD10_chap_XIV          |       58201 |            0 |             87.23 |
| ICD10_chap_XIII         |       58201 |            0 |             87.23 |
| ICD10_chap_XII          |       58201 |            0 |             87.23 |
| ICD10_chap_XI           |       58201 |            0 |             87.23 |
| ICD10_chap_X            |       58201 |            0 |             87.23 |
| ICD10_chap_IX           |       58201 |            0 |             87.23 |
| ICD10_chap_VIII         |       58201 |            0 |             87.23 |
| ICD10_chap_XX           |       58201 |            0 |             87.23 |
| ICD10_chap_XVII         |       58201 |            0 |             87.23 |
| ICD10_chap_XVIII        |       58201 |            0 |             87.23 |
| ICD10_chap_XIX          |       58201 |            0 |             87.23 |
| ICD10_chap_XXII         |       58201 |            0 |             87.23 |
| ICD10_chap_XXI          |       58201 |            0 |             87.23 |
| ICD10_chap_I            |       58201 |            0 |             87.23 |
| ICD10_chap_VI           |       58201 |            0 |             87.23 |
| ICD10_chap_II           |       58201 |            0 |             87.23 |
| ICD10_chap_III          |       58201 |            0 |             87.23 |
| ICD10_chap_IV           |       58201 |            0 |             87.23 |
| ICD10_chap_VII          |       58201 |            0 |             87.23 |
| ICD10_chap_V            |       58201 |            0 |             87.23 |
| VITAL_temperature_std   |       56990 |            0 |             85.42 |
| VITAL_temperature_min   |       56940 |            1 |             85.34 |
| VITAL_temperature_max   |       56940 |            0 |             85.34 |
| VITAL_temperature_mean  |       56940 |            0 |             85.34 |
| LAB_troponin__I_std     |       55475 |            0 |             83.15 |
| LAB_lactate_std         |       53849 |            0 |             80.71 |
| VITAL_cvp_min           |       51654 |          842 |             78.68 |
| VITAL_cvp_std           |       51774 |            0 |             77.6  |
| VITAL_cvp_max           |       51654 |            6 |             77.43 |
| VITAL_cvp_mean          |       51654 |            4 |             77.43 |
| LAB_troponin__I_max     |       50757 |            0 |             76.08 |
| LAB_troponin__I_mean    |       50757 |            0 |             76.08 |
| LAB_troponin__I_min     |       50757 |            0 |             76.08 |
| LAB_PT__INR_std         |       50003 |            0 |             74.95 |
| LAB_phosphate_std       |       45765 |            0 |             68.59 |
| APACHE_fio2             |           0 |        44165 |             66.2  |
| APACHE_pco2             |           0 |        44165 |             66.2  |
| APACHE_pao2             |           0 |        44165 |             66.2  |
| APACHE_ph               |           0 |        44165 |             66.2  |
| LAB_lactate_min         |       42670 |            0 |             63.95 |
| LAB_lactate_max         |       42670 |            0 |             63.95 |
| LAB_lactate_mean        |       42670 |            0 |             63.95 |
| APACHE_bilirubin        |           0 |        38707 |             58.01 |
| APACHE_albumin          |           0 |        36008 |             53.97 |
| LAB_PT__INR_mean        |       34100 |            0 |             51.11 |
| LAB_PT__INR_min         |       34100 |            0 |             51.11 |
| LAB_PT__INR_max         |       34100 |            0 |             51.11 |
| LAB_magnesium_std       |       33557 |            0 |             50.3  |
| APACHE_urine            |           0 |        32989 |             49.44 |
| LAB_phosphate_mean      |       32922 |            0 |             49.34 |
| LAB_phosphate_max       |       32922 |            0 |             49.34 |
| LAB_phosphate_min       |       32922 |            0 |             49.34 |
| TREAT_vasopressor       |       24810 |            0 |             37.19 |
| TREAT_inotropes         |       24810 |            0 |             37.19 |
| TREAT_mechanical_vent   |       24810 |            0 |             37.19 |
| TREAT_antiarrhythmics   |       24810 |            0 |             37.19 |
| TREAT_iv_fluids         |       24810 |            0 |             37.19 |
| TREAT_airway_mgmt       |       24810 |            0 |             37.19 |
| TREAT_neuro_therapy     |       24810 |            0 |             37.19 |
| TREAT_beta_blockers     |       24810 |            0 |             37.19 |
| TREAT_ace_inhibitors    |       24810 |            0 |             37.19 |
| TREAT_surgery           |       24810 |            0 |             37.19 |
| TREAT_vascular_access   |       24810 |            0 |             37.19 |
| TREAT_antibiotics       |       24810 |            0 |             37.19 |
| TREAT_gi_prophylaxis    |       24810 |            0 |             37.19 |
| TREAT_enteral_nutrition |       24810 |            0 |             37.19 |
| TREAT_transfusion       |       24810 |            0 |             37.19 |
| TREAT_insulin           |       24810 |            0 |             37.19 |
| TREAT_dialysis          |       24810 |            0 |             37.19 |
| TREAT_analgesia         |       24810 |            0 |             37.19 |
| TREAT_sedation          |       24810 |            0 |             37.19 |
| TREAT_toxicology        |       24810 |            0 |             37.19 |
| LAB_magnesium_max       |       19065 |            0 |             28.58 |
| LAB_magnesium_mean      |       19065 |            0 |             28.58 |
| LAB_magnesium_min       |       19065 |            0 |             28.58 |
| APACHE_wbc              |           0 |        12028 |             18.03 |
| APACHE_hematocrit       |           0 |        10686 |             16.02 |
| APACHE_bun              |           0 |        10099 |             15.14 |
| APACHE_creatinine       |           0 |         9876 |             14.8  |
| APACHE_sodium           |           0 |         9637 |             14.44 |
| LAB_potassium_std       |        6360 |            0 |              9.53 |
| VITAL_respiration_std   |        4899 |            0 |              7.34 |
| APACHE_glucose          |           0 |         4861 |              7.29 |
| VITAL_respiration_mean  |        4862 |            0 |              7.29 |
| VITAL_respiration_max   |        4862 |            0 |              7.29 |
| VITAL_respiration_min   |        4862 |            0 |              7.29 |
| APACHE_temperature      |           0 |         2997 |              4.49 |
| LAB_potassium_mean      |        1420 |            0 |              2.13 |
| LAB_potassium_max       |        1420 |            0 |              2.13 |
| LAB_potassium_min       |        1420 |            0 |              2.13 |
| VITAL_sao2_std          |        1095 |            0 |              1.64 |
| VITAL_sao2_min          |        1075 |            0 |              1.61 |
| VITAL_sao2_mean         |        1075 |            0 |              1.61 |
| VITAL_sao2_max          |        1075 |            0 |              1.61 |
| VITAL_heartrate_std     |         653 |            0 |              0.98 |
| VITAL_heartrate_mean    |         651 |            0 |              0.98 |
| VITAL_heartrate_max     |         651 |            0 |              0.98 |
| VITAL_heartrate_min     |         651 |            0 |              0.98 |
| APACHE_respiratoryrate  |           0 |          338 |              0.51 |
| APACHE_meds             |           0 |          299 |              0.45 |
| APACHE_meanbp           |           0 |          171 |              0.26 |
| APACHE_heartrate        |           0 |          129 |              0.19 |
| gender_male             |          13 |            0 |              0.02 |

## 3. Duplicates
- **Duplicate rows**: 0

## 4. Target and Survival Analysis
- **mortality distribution**:
  - 0: 93.20%
  - 1: 6.80%
- **survival_hours range**: [48.0, 719.5]
### Clinical Sanity Checks
- **Invalid ages (<0 or >120)**: 0
- **Patients with very short stay and mortality (<= 4h)**: 0

## 5. Numerical Column Issues
### Constant and Near-Constant Columns
#### Constant/Near-Constant (non-empty)
|    | Column                | Status   |   Value |   % Matching |
|---:|:----------------------|:---------|--------:|-------------:|
|  0 | TREAT_vasopressor     | Constant |       0 |          100 |
|  1 | TREAT_antibiotics     | Constant |       0 |          100 |
|  2 | TREAT_sedation        | Constant |       0 |          100 |
|  3 | TREAT_analgesia       | Constant |       0 |          100 |
|  4 | TREAT_insulin         | Constant |       0 |          100 |
|  5 | TREAT_transfusion     | Constant |       0 |          100 |
|  6 | TREAT_gi_prophylaxis  | Constant |       0 |          100 |
|  7 | TREAT_vascular_access | Constant |       0 |          100 |
|  8 | TREAT_antiarrhythmics | Constant |       0 |          100 |
|  9 | TREAT_iv_fluids       | Constant |       0 |          100 |
| 10 | TREAT_airway_mgmt     | Constant |       0 |          100 |
| 11 | TREAT_neuro_therapy   | Constant |       0 |          100 |
| 12 | TREAT_beta_blockers   | Constant |       0 |          100 |
| 13 | TREAT_ace_inhibitors  | Constant |       0 |          100 |
| 14 | TREAT_inotropes       | Constant |       0 |          100 |
| 15 | ICD10_chap_I          | Constant |       0 |          100 |
| 16 | ICD10_chap_III        | Constant |       0 |          100 |
| 17 | ICD10_chap_VII        | Constant |       0 |          100 |
| 18 | ICD10_chap_VIII       | Constant |       0 |          100 |
| 19 | ICD10_chap_XII        | Constant |       0 |          100 |
| 20 | ICD10_chap_XIV        | Constant |       0 |          100 |
| 21 | ICD10_chap_XVI        | Constant |       0 |          100 |
| 22 | ICD10_chap_XVII       | Constant |       0 |          100 |
| 23 | ICD10_chap_XX         | Constant |       0 |          100 |
| 24 | ICD10_chap_XXI        | Constant |       0 |          100 |
| 25 | ICD10_chap_XXII       | Constant |       0 |          100 |

### Potential Outliers (Z-score > 3)
|    | Column                  |   Outliers |   % Outliers |
|---:|:------------------------|-----------:|-------------:|
| 17 | APACHE_motor_1          |       5228 |         7.84 |
| 20 | APACHE_motor_4          |       5106 |         7.65 |
| 16 | APACHE_eyes_2           |       4767 |         7.14 |
| 66 | TREAT_mechanical_vent   |       3613 |         8.62 |
| 67 | TREAT_dialysis          |       3061 |         7.3  |
| 22 | APACHE_verbal_3         |       3005 |         4.5  |
| 68 | TREAT_enteral_nutrition |       2752 |         6.57 |
|  0 | APACHE_dialysis         |       2598 |         3.89 |
| 78 | eth_hispanic            |       2453 |         3.68 |
|  4 | APACHE_temperature      |       1744 |         2.74 |
| 21 | APACHE_verbal_2         |       1719 |         2.58 |
| 23 | VITAL_heartrate_min     |       1625 |         2.46 |
| 31 | VITAL_sao2_min          |       1476 |         2.25 |
|  9 | APACHE_creatinine       |       1378 |         2.42 |
| 39 | VITAL_respiration_max   |       1314 |         2.12 |
| 13 | APACHE_bun              |       1134 |         2    |
| 79 | eth_asian               |       1086 |         1.63 |
|  1 | APACHE_meds             |       1023 |         1.54 |
| 47 | LAB_potassium_max       |        968 |         1.48 |
|  5 | APACHE_sodium           |        941 |         1.65 |
| 32 | VITAL_sao2_max          |        907 |         1.38 |
| 14 | APACHE_glucose          |        901 |         1.46 |
| 34 | VITAL_sao2_std          |        875 |         1.33 |
| 49 | LAB_potassium_std       |        870 |         1.44 |
| 41 | VITAL_respiration_std   |        867 |         1.4  |
| 26 | VITAL_heartrate_std     |        856 |         1.3  |
| 69 | TREAT_toxicology        |        817 |         1.95 |
| 52 | LAB_PT__INR_mean        |        782 |         2.4  |
| 51 | LAB_PT__INR_max         |        740 |         2.27 |
| 75 | ICD10_chap_XI           |        697 |         8.18 |
|  3 | APACHE_wbc              |        693 |         1.27 |
| 50 | LAB_PT__INR_min         |        664 |         2.04 |
| 40 | VITAL_respiration_mean  |        616 |         1    |
| 19 | APACHE_motor_3          |        615 |         0.92 |
| 11 | APACHE_pao2             |        587 |         2.6  |
| 36 | VITAL_cvp_mean          |        577 |         3.83 |
| 48 | LAB_potassium_mean      |        571 |         0.87 |
| 43 | LAB_lactate_max         |        564 |         2.35 |
| 63 | LAB_phosphate_max       |        563 |         1.67 |
| 60 | LAB_magnesium_mean      |        537 |         1.13 |
| 64 | LAB_phosphate_mean      |        533 |         1.58 |
|  2 | APACHE_urine            |        532 |         1.58 |
| 44 | LAB_lactate_mean        |        510 |         2.12 |
| 59 | LAB_magnesium_max       |        494 |         1.04 |
| 62 | LAB_phosphate_min       |        484 |         1.43 |
| 24 | VITAL_heartrate_max     |        442 |         0.67 |
| 15 | APACHE_bilirubin        |        438 |         1.56 |
| 42 | LAB_lactate_min         |        428 |         1.78 |
| 61 | LAB_magnesium_std       |        412 |         1.24 |
| 58 | LAB_magnesium_min       |        410 |         0.86 |
| 46 | LAB_potassium_min       |        409 |         0.63 |
| 12 | APACHE_pco2             |        403 |         1.79 |
| 33 | VITAL_sao2_mean         |        385 |         0.59 |
| 18 | APACHE_motor_2          |        366 |         0.55 |
| 53 | LAB_PT__INR_std         |        350 |         2.09 |
| 65 | LAB_phosphate_std       |        338 |         1.61 |
| 45 | LAB_lactate_std         |        271 |         2.11 |
|  8 | APACHE_hematocrit       |        244 |         0.44 |
| 54 | LAB_troponin__I_min     |        236 |         1.48 |
| 28 | VITAL_temperature_max   |        229 |         2.34 |
| 29 | VITAL_temperature_mean  |        224 |         2.29 |
| 56 | LAB_troponin__I_mean    |        218 |         1.37 |
| 55 | LAB_troponin__I_max     |        213 |         1.33 |
|  7 | APACHE_ph               |        202 |         0.9  |
| 25 | VITAL_heartrate_mean    |        177 |         0.27 |
| 35 | VITAL_cvp_min           |        126 |         0.89 |
| 57 | LAB_troponin__I_std     |        118 |         1.05 |
|  6 | APACHE_heartrate        |        106 |         0.16 |
| 70 | ICD10_chap_II           |        100 |         1.17 |
| 72 | ICD10_chap_V            |         93 |         1.09 |
| 76 | ICD10_chap_XIII         |         88 |         1.03 |
| 37 | VITAL_cvp_std           |         82 |         0.55 |
| 10 | APACHE_albumin          |         48 |         0.16 |
| 38 | VITAL_respiration_min   |         47 |         0.08 |
| 73 | ICD10_chap_VI           |         43 |         0.5  |
| 74 | ICD10_chap_IX           |         23 |         0.27 |
| 71 | ICD10_chap_IV           |         21 |         0.25 |
| 30 | VITAL_temperature_std   |         17 |         0.17 |
| 27 | VITAL_temperature_min   |          5 |         0.05 |
| 77 | ICD10_chap_XV           |          4 |         0.05 |

## 6. Correlation Analysis
### Highly Collinear Features (> 0.9)
|    | Var1                      | Var2                      |   Correlation |
|---:|:--------------------------|:--------------------------|--------------:|
|  0 | patienthealthsystemstayid | patientunitstayid         |        0.9992 |
|  1 | hospitalid                | patientunitstayid         |        0.9952 |
|  2 | hospitalid                | patienthealthsystemstayid |        0.9944 |
|  3 | VITAL_temperature_mean    | VITAL_temperature_max     |        0.982  |
|  4 | VITAL_temperature_std     | VITAL_temperature_min     |       -0.9303 |
|  5 | LAB_lactate_mean          | LAB_lactate_max           |        0.9368 |
|  6 | LAB_PT__INR_mean          | LAB_PT__INR_min           |        0.9135 |
|  7 | LAB_PT__INR_mean          | LAB_PT__INR_max           |        0.942  |
|  8 | LAB_troponin__I_mean      | LAB_troponin__I_min       |        0.9194 |
|  9 | LAB_troponin__I_mean      | LAB_troponin__I_max       |        0.9567 |
| 10 | LAB_troponin__I_std       | LAB_troponin__I_max       |        0.9451 |
| 11 | LAB_phosphate_mean        | LAB_phosphate_min         |        0.925  |
| 12 | LAB_phosphate_mean        | LAB_phosphate_max         |        0.9207 |

### Top Correlations with mortality
|                   |   mortality |
|:------------------|------------:|
| mortality         |    1        |
| LAB_lactate_mean  |    0.264389 |
| LAB_lactate_max   |    0.255352 |
| LAB_lactate_min   |    0.231821 |
| APACHE_eyes_1     |    0.169915 |
| APACHE_motor_1    |    0.160385 |
| LAB_lactate_std   |    0.155911 |
| APACHE_verbal_1   |    0.141879 |
| APACHE_vent       |    0.13572  |
| LAB_phosphate_max |    0.128985 |

|                 |   mortality |
|:----------------|------------:|
| ICD10_chap_III  |         nan |
| ICD10_chap_VII  |         nan |
| ICD10_chap_VIII |         nan |
| ICD10_chap_XII  |         nan |
| ICD10_chap_XIV  |         nan |
| ICD10_chap_XVI  |         nan |
| ICD10_chap_XVII |         nan |
| ICD10_chap_XX   |         nan |
| ICD10_chap_XXI  |         nan |
| ICD10_chap_XXII |         nan |

## 7. Categorical Column Analysis
|    | Column    |   Unique Values | Top Value   |
|---:|:----------|----------------:|:------------|
|  0 | uniquepid |           59675 | 030-184     |

## 8. Bin Count (Unique Values / Discretization)
Unique values count for numeric columns (high unique = continuous, low unique = ordinal/binned):

|     | Column                  |   Unique Values |
|----:|:------------------------|----------------:|
|  96 | TREAT_neuro_therapy     |               1 |
|  97 | TREAT_beta_blockers     |               1 |
|  98 | TREAT_ace_inhibitors    |               1 |
| 101 | TREAT_inotropes         |               1 |
| 102 | ICD10_chap_I            |               1 |
| 109 | ICD10_chap_VIII         |               1 |
|  87 | TREAT_analgesia         |               1 |
| 108 | ICD10_chap_VII          |               1 |
| 117 | ICD10_chap_XVI          |               1 |
| 121 | ICD10_chap_XX           |               1 |
| 123 | ICD10_chap_XXII         |               1 |
| 122 | ICD10_chap_XXI          |               1 |
| 115 | ICD10_chap_XIV          |               1 |
| 118 | ICD10_chap_XVII         |               1 |
| 113 | ICD10_chap_XII          |               1 |
| 104 | ICD10_chap_III          |               1 |
|  94 | TREAT_antiarrhythmics   |               1 |
|  95 | TREAT_airway_mgmt       |               1 |
|  85 | TREAT_antibiotics       |               1 |
|  84 | TREAT_vasopressor       |               1 |
|  92 | TREAT_gi_prophylaxis    |               1 |
|  93 | TREAT_vascular_access   |               1 |
|  90 | TREAT_transfusion       |               1 |
|  86 | TREAT_sedation          |               1 |
|  89 | TREAT_insulin           |               1 |
|   1 | mortality               |               2 |
|   5 | APACHE_dialysis         |               2 |
|   3 | APACHE_intubated        |               2 |
|  28 | APACHE_motor_1          |               2 |
|  29 | APACHE_motor_2          |               2 |
|  30 | APACHE_motor_3          |               2 |
|   4 | APACHE_vent             |               2 |
| 111 | ICD10_chap_X            |               2 |
| 110 | ICD10_chap_IX           |               2 |
| 106 | ICD10_chap_V            |               2 |
|  88 | TREAT_dialysis          |               2 |
|  91 | TREAT_enteral_nutrition |               2 |
|  83 | TREAT_mechanical_vent   |               2 |
| 103 | ICD10_chap_II           |               2 |
|  31 | APACHE_motor_4          |               2 |
|  24 | APACHE_eyes_1           |               2 |
|  25 | APACHE_eyes_2           |               2 |
|  26 | APACHE_eyes_3           |               2 |
|  27 | APACHE_eyes_4           |               2 |
|  35 | APACHE_verbal_2         |               2 |
|  34 | APACHE_verbal_1         |               2 |
|  33 | APACHE_motor_6          |               2 |
|  32 | APACHE_motor_5          |               2 |
|  37 | APACHE_verbal_4         |               2 |
|  38 | APACHE_verbal_5         |               2 |
|  36 | APACHE_verbal_3         |               2 |
| 127 | eth_hispanic            |               2 |
| 120 | ICD10_chap_XIX          |               2 |
| 126 | eth_african_american    |               2 |
| 125 | eth_caucasian           |               2 |
| 124 | gender_male             |               2 |
| 114 | ICD10_chap_XIII         |               2 |
| 100 | TREAT_toxicology        |               2 |
|  99 | TREAT_surgery           |               2 |
| 105 | ICD10_chap_IV           |               2 |
| 107 | ICD10_chap_VI           |               2 |
| 112 | ICD10_chap_XI           |               2 |
| 116 | ICD10_chap_XV           |               2 |
| 119 | ICD10_chap_XVIII        |               2 |
| 128 | eth_asian               |               2 |
|   6 | APACHE_meds             |               3 |
|  48 | VITAL_sao2_max          |              18 |
|  55 | VITAL_respiration_min   |              45 |
|  17 | APACHE_albumin          |              53 |
|  10 | APACHE_respiratoryrate  |              73 |
|   2 | age                     |              73 |
|  23 | APACHE_fio2             |              84 |
|  47 | VITAL_sao2_min          |             101 |
|  63 | LAB_potassium_min       |             106 |
|  64 | LAB_potassium_max       |             114 |
|  79 | LAB_phosphate_min       |             132 |
|  39 | VITAL_heartrate_min     |             136 |
|  11 | APACHE_sodium           |             155 |
|  80 | LAB_phosphate_max       |             155 |
|  13 | APACHE_meanbp           |             169 |
|  56 | VITAL_respiration_max   |             181 |
|  51 | VITAL_cvp_min           |             197 |
|  12 | APACHE_heartrate        |             200 |
|  40 | VITAL_heartrate_max     |             206 |
|  75 | LAB_magnesium_min       |             235 |
|  76 | LAB_magnesium_max       |             284 |
|   9 | APACHE_temperature      |             326 |
|  44 | VITAL_temperature_max   |             333 |
|  52 | VITAL_cvp_max           |             398 |
|  67 | LAB_PT__INR_min         |             411 |
|  59 | LAB_lactate_min         |             450 |
|  20 | APACHE_bun              |             455 |
|  22 | APACHE_bilirubin        |             504 |
|  15 | APACHE_hematocrit       |             516 |
|  68 | LAB_PT__INR_max         |             601 |
|  14 | APACHE_ph               |             604 |
|  60 | LAB_lactate_max         |             767 |
|  19 | APACHE_pco2             |             833 |
|  21 | APACHE_glucose          |             877 |
|  43 | VITAL_temperature_min   |             893 |
|  77 | LAB_magnesium_mean      |            1133 |
|  16 | APACHE_creatinine       |            1341 |
|  81 | LAB_phosphate_mean      |            1457 |
|  69 | LAB_PT__INR_mean        |            1931 |
|  18 | APACHE_pao2             |            1999 |
|  65 | LAB_potassium_mean      |            2151 |
|  71 | LAB_troponin__I_min     |            2407 |
|  61 | LAB_lactate_mean        |            2700 |
|  72 | LAB_troponin__I_max     |            3006 |
|   8 | APACHE_wbc              |            3153 |
|  78 | LAB_magnesium_std       |            3393 |
|  70 | LAB_PT__INR_std         |            3569 |
|  82 | LAB_phosphate_std       |            4299 |
|  73 | LAB_troponin__I_mean    |            5247 |
|  74 | LAB_troponin__I_std     |            5513 |
|  62 | LAB_lactate_std         |            5665 |
|  45 | VITAL_temperature_mean  |            9669 |
|  46 | VITAL_temperature_std   |            9675 |
|  66 | LAB_potassium_std       |           10939 |
|  53 | VITAL_cvp_mean          |           14641 |
|  54 | VITAL_cvp_std           |           14859 |
|   0 | survival_hours          |           15421 |
|   7 | APACHE_urine            |           22250 |
|  57 | VITAL_respiration_mean  |           56047 |
|  49 | VITAL_sao2_mean         |           56343 |
|  58 | VITAL_respiration_std   |           61781 |
|  41 | VITAL_heartrate_mean    |           63012 |
|  50 | VITAL_sao2_std          |           65473 |
|  42 | VITAL_heartrate_std     |           66043 |

