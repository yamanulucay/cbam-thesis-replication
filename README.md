# CBAM Thesis — Replication

Replication code and data for the undergraduate thesis **"CBAM-Related Transition Activity and Short-Term Financial Performance: Evidence from Turkish CBAM-Covered Sectors"** (Yaman Uluçay, Yıldız Technical University).

The analysis was carried out in **jamovi (v2.7)**. This repository also provides a **Python script** that independently re-estimates both regression models and reproduces every coefficient, fit measure, and diagnostic reported in the thesis. Running it prints a line-by-line check confirming the Python output matches the jamovi output.

## Files

| File | Description |
|------|-------------|
| `cbam_replication.py` | Replication script. Reads the dataset below. |
| `cbam_jamovi_input_101rows_with_ebitda_w.xlsx` | Dataset used by the script (101 firm-year obs.; includes original and winsorized EBITDA margin). |
| `ebitda_regression_cbam v2.omv` | Original jamovi file — main model. |
| `ebitda_regression_cbam_w v2.omv` | Original jamovi file — robustness model. |

## Requirements

Python 3.8 or later (tested on 3.12). Install the libraries:

```bash
pip install pandas numpy scipy statsmodels openpyxl
```

## How to run

**Google Colab (no installation):** create a new notebook and run:

```python
!git clone https://github.com/yamanulucay/cbam-thesis-replication.git
%cd cbam-thesis-replication
!pip install -q statsmodels openpyxl
!python cbam_replication.py
```

**Local computer:** download or clone the repository, keep `cbam_replication.py` and the `.xlsx` file in the same folder, then run:

```bash
python cbam_replication.py
```

## Output

The script estimates the main model (EBITDA margin) and the robustness model (winsorized EBITDA margin + year controls), runs the diagnostic tests, and prints a consistency check. When set up correctly, every value is marked `OK`:

```
Quantity                              Computed    Thesis   Match?
Main  - CBAM Activity Score beta       -0.0325   -0.0325       OK
Main  - CBAM Activity Score p            0.0134    0.0130      OK
Robust- CBAM Activity Score beta       -0.0187   -0.0187       OK
Robust- CBAM Activity Score p            0.0771    0.0770      OK
...
```

(Small differences such as `0.0134` vs `0.013` are rounding only.)
