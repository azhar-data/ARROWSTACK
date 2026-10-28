# Azhar ke liye step-by-step Python guide

## 1. Dono projects ka simple matlab

**Task 1 — EDA:** data ko samajhna: rows, missing values, averages aur graphs.
Is task mein machine-learning model train nahi hota.

**Task 2 — Regression:** car ki specifications se fuel efficiency `mpg` predict
karni hai. `mpg` continuous number hai, is liye regression use hoti hai.

## 2. Project download karo

Repository: [azhar-data/ARROWSTACK](https://github.com/azhar-data/ARROWSTACK).

```powershell
git clone https://github.com/azhar-data/ARROWSTACK.git
cd ARROWSTACK
```

Ya GitHub par **Code → Download ZIP** karo, extract karo, aur VS Code mein folder
open karo. Terminal mein `run_all.py` aur `requirements.txt` wali directory honi chahiye.

## 3. Python environment banao

Windows PowerShell (Python 3.12):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Dono projects run karo

```powershell
.\.venv\Scripts\python.exe run_all.py
```

Is se EDA reports/graphs aur regression model/results regenerate honge. Data
repository mein included hai, is liye dataset download ya Kaggle account ki zaroorat nahi.

## 5. Task 1 samjho

File: `task1_eda/analysis.py`.

| Function | Kya karta hai |
| --- | --- |
| `load_data()` | Original CSV read karta hai |
| `clean_data()` | Columns validate aur labels normalize karta hai |
| `summarize()` | Species-wise counts aur averages banata hai |
| `correlations()` | Measurements ke relationships nikalta hai |
| `outlier_review()` | Unusual values ko review ke liye flag karta hai |
| `run()` | Tables, figures aur report save karta hai |

Missing measurements ko bina wajah mean se replace karna analysis badal sakta hai.
Is project mein EDA ke missing values preserve kiye jate hain. Correlation causation
ka proof nahi hoti; species aur island ka effect mix ho sakta hai.

## 6. Task 2 mein X aur y

File: `task2_regression/train.py`.

- `X`: cylinders, displacement, horsepower, weight, acceleration, model_year aur origin.
- `y = mpg`: woh number jo model predict karta hai.
- `car_name`: sirf group splitting ke liye; model input nahi.

Pipeline numeric missing values ko impute karti hai, features scale karti hai aur
origin ko one-hot encode karti hai. Preprocessing training folds ke andar fit hoti
hai taake test data ki information training mein leak na ho.

## 7. Metrics ka matlab

| Metric | Matlab | Behtar direction |
| --- | --- | --- |
| MAE | Average absolute error, mpg mein | Kam |
| MSE | Squared errors ka average | Kam |
| RMSE | MSE ka square root | Kam |
| R² | Variance-based predictive score | Aam tor par zyada |

Recorded holdout RMSE lagbhag **2.69 mpg** aur R² **0.885** hai. R² ka matlab
88.5% classification accuracy nahi. Dataset historical 1970–1982 vehicles ka hai,
is liye isay modern cars par validated model na samjho.

## 8. Example prediction

Training pehle run karo, phir:

```powershell
.\.venv\Scripts\python.exe -m task2_regression.predict --input task2_regression/examples/new_cars.csv --output my_predictions.csv
```

Output mein `predicted_mpg` column hoga. `model_year` source coding mein 70 se
82 tak hota hai. Existing output file ko overwrite nahi kiya jata.

## 9. Tests aur deliverable checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_deliverables.py
```

Tests data integrity, missingness, train/test group separation, preprocessing aur
prediction pipeline check karte hain. Passing tests internship approval ki guarantee nahi.

## 10. Common errors

| Error | Kya check karo |
| --- | --- |
| `ModuleNotFoundError` | Requirements same `.venv` Python se install hain? |
| `No module named task1_eda` | Terminal repository root mein hai? |
| `Model not found` | Pehle `run_all.py` ya regression training run karo |
| Missing feature columns | Input CSV mein expected column names hain? |
| Output already exists | Naya output filename use karo |

## Submission se pehle

Official assignment rubric se scope confirm karo. Dono Python projects ko samjho,
khud run karo aur findings apni zubaan mein explain karne ki practice karo. Assistance
disclosure required ho to internship policy follow karo.
