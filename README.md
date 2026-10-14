# Sepsis Diagnostic Analysis — Team Handoff

## Files
- `src/diagnostic_analysis.py`: reusable analysis module; integrate this into the group's shared project.
- `app.py`: standalone Streamlit dashboard.
- `run_analysis.py`: command-line runner that exports tables and figures.
- `requirements.txt`: Python dependencies.

## Input columns
`age_years`; `sex_0male_1female` (0 Male, 1 Female); `episode_number` (expected 1–5);
`hospital_outcome_1alive_0dead` (0 Dead, 1 Alive).

## Run Streamlit on Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
If activation is blocked:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open the local URL shown in the terminal, usually http://localhost:8501.

## Run analysis without Streamlit
```powershell
python run_analysis.py --input "sepsis_survival_primary_cohort.csv"
```
Outputs go to `analysis_output/`.

## Integrate into the team's app
Copy `src/diagnostic_analysis.py` into the combined project's `src/` folder and import
`run_diagnostic_analysis`. Do not overwrite the group's main `app.py`; add your work as a
Diagnostic tab/section and let the app display the returned tables and figures.

## Important interpretation notes
- Invalid codes and negative ages are marked missing; analyses use available cases.
- Regression uses complete cases and episode 1 as the reference category.
- ROC-AUC, Brier score, and calibration are calculated on model-fitting data (in-sample),
  not held-out validation.
- If rows repeat patients, independence may be violated; discuss patient-cluster-aware
  methods if patient identifiers are available.
- This is an educational/research analysis, not a clinically validated prediction tool.
