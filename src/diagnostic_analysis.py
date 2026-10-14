
"""Reusable diagnostic analysis for the sepsis survival cohort."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

REQUIRED = ["age_years", "sex_0male_1female", "episode_number",
            "hospital_outcome_1alive_0dead"]
EPISODES = [1, 2, 3, 4, 5]
sns.set_theme(style="whitegrid")


def prepare_data(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    d = df.copy()
    for c in REQUIRED:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d.loc[d.age_years < 0, "age_years"] = np.nan
    d.loc[~d["sex_0male_1female"].isin([0, 1]), "sex_0male_1female"] = np.nan
    d.loc[~d["hospital_outcome_1alive_0dead"].isin([0, 1]),
          "hospital_outcome_1alive_0dead"] = np.nan
    d.loc[~d.episode_number.isin(EPISODES), "episode_number"] = np.nan
    d["death"] = 1 - d["hospital_outcome_1alive_0dead"]
    d["sex_label"] = d["sex_0male_1female"].map({0: "Male", 1: "Female"})
    d["outcome_label"] = d["hospital_outcome_1alive_0dead"].map({0: "Dead", 1: "Alive"})
    quality = pd.DataFrame([
        ("Original rows", len(df)),
        ("Duplicate rows in original data", int(df.duplicated().sum())),
        *[(f"Missing/invalid: {c}", int(d[c].isna().sum())) for c in REQUIRED]
    ], columns=["Check", "Value"])
    return d, quality


def run_diagnostic_analysis(df):
    d, quality = prepare_data(df)
    desc = d[REQUIRED].describe().T.round(2)

    # Age by outcome
    ad = d.dropna(subset=["age_years", "outcome_label"])
    age_summary = ad.groupby("outcome_label")["age_years"].agg(
        ["count", "mean", "std", "median", "min", "max"]).reindex(["Alive", "Dead"])
    dead = ad.loc[ad.outcome_label == "Dead", "age_years"]
    alive = ad.loc[ad.outcome_label == "Alive", "age_years"]
    age_tests = []
    if len(dead) and len(alive):
        t = stats.ttest_ind(dead, alive, equal_var=False, nan_policy="omit")
        u = stats.mannwhitneyu(dead, alive, alternative="two-sided")
        age_tests = [{"Test": "Welch's t-test", "Statistic": t.statistic, "P-value": t.pvalue},
                     {"Test": "Mann–Whitney U", "Statistic": u.statistic, "P-value": u.pvalue}]
    age_tests = pd.DataFrame(age_tests)

    # Sex by outcome
    sd = d.dropna(subset=["sex_label", "outcome_label"])
    sex_table = pd.crosstab(sd.sex_label, sd.outcome_label).reindex(
        index=["Female", "Male"], columns=["Alive", "Dead"], fill_value=0)
    sex_pct = sex_table.div(sex_table.sum(axis=1).replace(0, np.nan), axis=0) * 100
    sex_tests = pd.DataFrame()
    if sex_table.values.sum():
        chi, p, dof, exp = stats.chi2_contingency(sex_table)
        sex_tests = pd.DataFrame([{"Test": "Chi-square", "Chi-square": chi, "df": dof,
                                   "P-value": p, "Minimum expected count": exp.min()}])

    # Episode by outcome
    ed = d.dropna(subset=["episode_number", "outcome_label"])
    episode_table = pd.crosstab(ed.episode_number, ed.outcome_label).reindex(
        index=EPISODES, columns=["Alive", "Dead"], fill_value=0)
    episode_rates = pd.DataFrame({"Total": episode_table.sum(axis=1),
                                  "Deaths": episode_table.Dead})
    episode_rates["Death (%)"] = np.where(
        episode_rates.Total > 0, 100 * episode_rates.Deaths / episode_rates.Total, np.nan)
    episode_tests = pd.DataFrame()
    if episode_table.values.sum():
        chi, p, dof, exp = stats.chi2_contingency(episode_table)
        episode_tests = pd.DataFrame([{"Test": "Chi-square", "Chi-square": chi, "df": dof,
                                       "P-value": p, "Minimum expected count": exp.min()}])

    # Complete-case multivariable logistic regression
    md = d.dropna(subset=["death", "age_years", "sex_0male_1female", "episode_number"]).copy()
    md = md[md.episode_number.isin(EPISODES)].copy()
    md["male"] = md["sex_0male_1female"].map({0: 1, 1: 0})
    md["episode_number"] = pd.Categorical(md.episode_number, categories=EPISODES)
    if md.death.nunique() < 2:
        raise ValueError("Logistic regression requires both deaths and survivors.")
    formula = "death ~ age_years + male + C(episode_number, Treatment(reference=1))"
    model = smf.logit(formula, data=md).fit(disp=False, maxiter=200)
    ci = model.conf_int()
    ors = pd.DataFrame({"Adjusted OR": np.exp(model.params),
                        "95% CI lower": np.exp(ci[0]), "95% CI upper": np.exp(ci[1]),
                        "p-value": model.pvalues})
    y = md.death.astype(int)
    prob = pd.Series(model.predict(md), index=md.index)
    metrics = {"n_observations": len(md), "deaths": int(y.sum()),
               "survivors": int((y == 0).sum()), "observed_mortality": float(y.mean()),
               "mean_predicted_mortality": float(prob.mean()),
               "roc_auc_in_sample": float(roc_auc_score(y, prob)),
               "brier_score_in_sample": float(brier_score_loss(y, prob))}

    # Figures
    figs = {}
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=d, x="age_years", bins=30, ax=ax)
    ax.set_title("Age distribution"); figs["age_histogram"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=d, x="age_years", ax=ax)
    ax.set_title("Age and potential outliers"); figs["age_boxplot"] = fig

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["sex_0male_1female", "hospital_outcome_1alive_0dead", "episode_number"]):
        sns.countplot(data=d, x=col, ax=ax)
        ax.set_title(col)
    fig.tight_layout(); figs["categorical_counts"] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=ad, x="outcome_label", y="age_years", order=["Alive", "Dead"],
                showfliers=False, ax=ax)
    ax.set_title("Age by hospital outcome"); figs["age_by_outcome"] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    sex_pct[["Alive", "Dead"]].plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylim(0, 100); ax.set_ylabel("Patients (%)"); ax.set_title("Outcome by sex")
    ax.tick_params(axis="x", rotation=0); fig.tight_layout(); figs["sex_outcome"] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=episode_rates.reset_index(), x="episode_number", y="Death (%)", ax=ax)
    ax.set_title("Mortality by episode"); figs["episode_mortality"] = fig

    plot = ors.drop(index="Intercept", errors="ignore").sort_values("Adjusted OR")
    fig, ax = plt.subplots(figsize=(9, max(4, len(plot) * .5)))
    ypos = np.arange(len(plot))
    ax.errorbar(plot["Adjusted OR"], ypos,
                xerr=[plot["Adjusted OR"] - plot["95% CI lower"],
                      plot["95% CI upper"] - plot["Adjusted OR"]], fmt="o", capsize=4)
    ax.axvline(1, linestyle="--"); ax.set_yticks(ypos); ax.set_yticklabels(plot.index)
    ax.set_xscale("log"); ax.set_xlabel("Adjusted OR (log scale)")
    ax.set_title("Multivariable logistic regression"); fig.tight_layout(); figs["forest_plot"] = fig

    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc_in_sample']:.3f}")
    ax.plot([0, 1], [0, 1], "--"); ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate"); ax.set_title("ROC curve"); ax.legend()
    figs["roc_curve"] = fig

    obs, pred = calibration_curve(y, prob, n_bins=min(10, max(2, len(y)//10)),
                                  strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pred, obs, "o-", label="Logistic regression"); ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed death proportion")
    ax.set_title("Calibration plot"); ax.legend(); figs["calibration_plot"] = fig

    return {
        "cleaned_data": d, "quality_report": quality, "descriptive_summary": desc,
        "age_summary": age_summary, "age_tests": age_tests,
        "sex_table": sex_table, "sex_percent": sex_pct, "sex_tests": sex_tests,
        "episode_table": episode_table, "episode_rates": episode_rates,
        "episode_tests": episode_tests, "model_data": md, "model": model,
        "odds_ratios": ors, "metrics": metrics, "figures": figs
    }
