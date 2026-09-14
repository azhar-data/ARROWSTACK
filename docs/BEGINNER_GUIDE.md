# Azhar ke liye step-by-step guide

## 1. Dono projects ka simple matlab

**Task 1 — EDA:** pehle data samajhna hai. Kitni rows hain? Kahan missing values
hain? Kaunsi species ka average weight zyada hai? Graphs kya relationship dikhate
hain? Is task mein model train nahi hota.

**Task 2 — Regression:** car ki specifications de kar uski fuel efficiency `mpg`
predict karni hai. `mpg` ek number hai, is liye regression use hoti hai.

## 2. Project download karo

GitHub repository: [azhar-data/ARROWSTACK](https://github.com/azhar-data/ARROWSTACK).

Git installed hai to terminal mein:

```powershell
git clone https://github.com/azhar-data/ARROWSTACK.git
cd ARROWSTACK
```

Git nahi hai to GitHub par **Code → Download ZIP** karo, ZIP extract karo, aur VS
Code mein extracted folder open karo. Terminal ka current folder woh hona chahiye
jahan `run_all.py` aur `requirements.txt` maujood hain.

## 3. Python environment banao

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.12 chahiye; agar `py -3.12` nahi milta to apni installed Python version
check karo. `.venv` is project ka separate environment hai. Direct Python path use
karne se PowerShell activation policy change karne ki zaroorat nahi.

## 4. Dono projects ek saath run karo

```powershell
.\.venv\Scripts\python.exe run_all.py
```

Is se EDA reports/graphs aur regression model/results regenerate ho jayenge.
Dataset repository mein hai, is liye dependency installation ke baad normal run
ke liye internet ya Kaggle account ki zaroorat nahi.

## 5. Task 1 samjho

File: `task1_eda/analysis.py`.

| Function | Kya karta hai |
| --- | --- |
| `load_data()` | Original Kaggle CSV read karta hai |
| `clean_data()` | Columns validate, labels normalize, missingness report karta hai |
| `summarize()` | Species-wise counts, averages aur uncertainty intervals banata hai |
| `correlations()` | Flipper aur mass ka pooled/within-species relationship nikalta hai |
| `outlier_review()` | Unusual values flag karta hai; delete nahi karta |
| `run()` | Saari tables, figures aur final report save karta hai |

Missing value ko EDA mein bina wajah mean se replace karna distribution badal
sakta hai. Is project mein missing measurements preserve hoti hain. Har statistic
available relevant records use karta hai. Complete-case result comparison bhi diya hai.

Important lesson: **correlation ka matlab causation nahi**. Agar Gentoo sirf ek
island par hai, island-wise difference species difference bhi ho sakta hai.

## 6. Task 2 mein X aur y

`X` model ko di jane wali information hai: cylinders, displacement, horsepower,
weight, acceleration, model_year aur origin.

`y = mpg` woh number hai jo predict karna hai.

`car_name` ko input nahi banaya. Isay sirf groups split karne ke liye use kiya
hai, taake exact same name training aur test dono mein na aa jaye.

## 7. Pipeline kya karti hai?

Numeric values ke liye median imputer missing cells handle karta hai aur scaler
features ko scale karta hai. Origin text ke liye imputer aur one-hot encoder use
hote hain. Us ke baad regression estimator fit hota hai.

Saari learned preprocessing training fold ke andar fit hoti hai. Test data ka
median ya scale training mein use nahi hota. Isi protection ko data leakage se
bachna kehte hain.

Cross-validation mein training data ke 5 group folds bante hain. Har dafa ek fold
validation ke liye aur baqi fitting ke liye use hotay hain. Lowest average RMSE
wala candidate select hota hai; final holdout selection mein use nahi hota.

## 8. Metrics ka simple matlab

| Metric | Matlab | Behtar direction |
| --- | --- | --- |
| MAE | Average absolute prediction error, mpg mein | Kam |
| MSE | Squared errors ka average | Kam |
| RMSE | MSE ka square root; bari mistakes ko zyada weight | Kam |
| R² | Variance-based predictive performance | Zyada, magar context zaroor dekho |

Is run mein holdout MAE lagbhag **1.94 mpg**, RMSE **2.69 mpg**, R² **0.885** hai.
**Iska matlab 88.5% classification accuracy nahi hai.** Dataset aur split badalne
se result badal sakta hai. Yeh purani cars ka educational benchmark hai.

## 9. Apni example prediction nikalo

Pehle training run honi chahiye. Phir:

```powershell
.\.venv\Scripts\python.exe -m task2_regression.predict --input task2_regression/examples/new_cars.csv --output my_predictions.csv
```

`my_predictions.csv` mein `predicted_mpg` column aa jayega. Yeh input rows practice
examples hain. `model_year` 70 se 82 tak source coding mein do; 2026 mat do.
Pehle se output file ho to naya filename do. Original input overwrite nahi hota.

## 10. Notebook open karo

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Jupyter mein `task1_eda/notebooks/penguins_eda.ipynb` ya
`task2_regression/notebooks/auto_mpg_regression.ipynb` open karo. VS Code notebook
mein `.venv` wala Python kernel select karo. Cells top-to-bottom run karo.
GitHub par saved outputs bina run kiye bhi dekh sakte ho.

## 11. Tests run karo

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

20 tests data hashes, missingness, train/test overlap, preprocessing, metrics aur
model reload check karte hain. `OK` ka matlab ye checks pass hain, official
internship evaluation pass hone ki guarantee nahi.

## 12. Common errors

| Error | Kya check karo |
| --- | --- |
| `ModuleNotFoundError` | Requirements same `.venv` Python se install ki hain? |
| `No module named task1_eda` | Terminal repository root mein hai? |
| `Model not found` | Pehle `run_all.py` ya regression training run karo |
| Missing required feature columns | Example CSV jaisi exact headings do |
| Output already exists | Naya output filename use karo |
| Source hash test fails | Raw source file edit/line-ending conversion to nahi hui? Original Git version compare karo |

## Submission se pehle

Official domain PDF ke exact projects confirm karo. Yeh EDA/Regression screenshot
titles ke mutabiq banay hain; Churn/Retail handbook tasks se alag hain. Notebook
ko samjho, khud run karo, aur apni zubaan mein findings explain karne ki practice
karo. Kisi assistance ki disclosure required ho to internship policy follow karo.
