from pathlib import Path
import io

import pandas as pd
import streamlit as st

from src.diagnostic_analysis import run_diagnostic_analysis


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Sepsis Diagnostic Analysis",
    page_icon="🩺",
    layout="wide",
)

st.title("Sepsis Diagnostic Analysis")
st.write(
    "This dashboard performs data quality checks, descriptive statistics, "
    " Diagnostic Data Analysis including- bivariate analyses, and multivariable logistic regression, and model evaluation."
)


# ==================================================
# 2. DATASET PATH
# ==================================================
PROJECT_DIR = Path(__file__).resolve().parent

# The app expects the CSV file inside the project's data folder.
# If Windows hides file extensions, confirm the actual extension is .csv.
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "sepsis_survival_primary_cohort.csv"
)


# ==================================================
# 3. VARIABLE INFORMATION
# ==================================================
with st.expander("Expected columns and coding"):
    st.markdown(
        """
        - `age_years`: age in years
        - `sex_0male_1female`: 0 = Male, 1 = Female
        - `episode_number`: expected categories 1–5
        - `hospital_outcome_1alive_0dead`: 0 = Dead, 1 = Alive
        - Regression outcome: `death = 1` for dead and `0` for alive
        """
    )


# ==================================================
# 4. AUTOMATICALLY LOAD DATASET
# ==================================================
if not DATA_PATH.is_file():
    st.error(
        "Dataset not found. Please check that the file exists at this location:"
    )
    st.code(str(DATA_PATH))
    st.info(
        "Place `sepsis_survival_primary_cohort.csv` inside the `data` folder "
        "next to the app.py file's project folder."
    )
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

st.success(f"Dataset loaded automatically from: data/{DATA_PATH.name}")
st.caption(f"Dataset size: {len(df):,} rows × {df.shape[1]:,} columns")

with st.expander("Preview first five rows"):
    st.dataframe(df.head(), use_container_width=True)


# ==================================================
# 5. RUN DIAGNOSTIC ANALYSIS
# ==================================================
try:
    with st.spinner("Running diagnostic analysis..."):
        results = run_diagnostic_analysis(df)

    # Create the five main tabs.
    tab_quality, tab_descriptive, tab_bivariate, tab_regression, tab_evaluation = (
        st.tabs(
            [
                "Data quality",
                "Descriptive",
                "Bivariate",
                "Regression",
                "Evaluation",
            ]
        )
    )

    # --------------------------------------------------
    # TAB 1: DATA QUALITY
    # --------------------------------------------------
    with tab_quality:
        st.header("Data quality inspection")

        st.subheader("Quality report")
        st.dataframe(
            results["quality_report"],
            use_container_width=True,
        )

        st.subheader("Missing values after validation")
        missing_report = pd.DataFrame(
            {
                "Missing count": results["cleaned_data"].isna().sum(),
                "Missing (%)": (
                    results["cleaned_data"].isna().mean() * 100
                ),
            }
        ).sort_values("Missing (%)", ascending=False)

        st.dataframe(
            missing_report.round(2),
            use_container_width=True,
        )

    # --------------------------------------------------
    # TAB 2: DESCRIPTIVE STATISTICS
    # --------------------------------------------------
    with tab_descriptive:
        st.header("Descriptive statistics")

        st.subheader("Numeric summary")
        st.dataframe(
            results["descriptive_summary"],
            use_container_width=True,
        )

        st.subheader("Age distribution")
        st.pyplot(
            results["figures"]["age_histogram"],
            clear_figure=False,
        )
        st.pyplot(
            results["figures"]["age_boxplot"],
            clear_figure=False,
        )

        st.subheader("Categorical variable distributions- - Sex, Episodes of Admission and Hospital Outcome")
        st.pyplot(
            results["figures"]["categorical_counts"],
            clear_figure=False,
        )

    # --------------------------------------------------
    # TAB 3: BIVARIATE ANALYSIS
    # --------------------------------------------------
    with tab_bivariate:
        st.header("Bivariate analysis")

        # Age versus outcome
        st.subheader("1. Age versus hospital outcome")
        st.write("Descriptive statistics by outcome")
        st.dataframe(
            results["age_summary"].round(2),
            use_container_width=True,
        )

        st.write("Welch's t-test and Mann–Whitney U test")
        st.dataframe(
            results["age_tests"].round(6),
            use_container_width=True,
        )

        st.pyplot(
            results["figures"]["age_by_outcome"],
            clear_figure=False,
        )

        st.divider()

        # Sex versus outcome
        st.subheader("2. Sex versus hospital outcome")
        st.write("Observed counts")
        st.dataframe(
            results["sex_table"],
            use_container_width=True,
        )

        st.write("Row percentages (%)")
        st.dataframe(
            results["sex_percent"].round(2),
            use_container_width=True,
        )

        st.write("Chi-square test")
        st.dataframe(
            results["sex_tests"].round(6),
            use_container_width=True,
        )

        st.pyplot(
            results["figures"]["sex_outcome"],
            clear_figure=False,
        )

        st.divider()

        # Episode versus outcome
        st.subheader("3. Episode number versus hospital outcome")
        st.write("Observed counts")
        st.dataframe(
            results["episode_table"],
            use_container_width=True,
        )

        st.write("Mortality by episode")
        st.dataframe(
            results["episode_rates"].round(2),
            use_container_width=True,
        )

        st.write("Chi-square test")
        st.dataframe(
            results["episode_tests"].round(6),
            use_container_width=True,
        )

        st.pyplot(
            results["figures"]["episode_mortality"],
            clear_figure=False,
        )

    # --------------------------------------------------
    # TAB 4: MULTIVARIABLE LOGISTIC REGRESSION
    # --------------------------------------------------
    with tab_regression:
        st.header("Multivariable logistic regression")

        st.caption(
            "Outcome: death (1 = dead, 0 = alive). "
            "Episode 1 is the reference category. "
            "The model adjusts for age, sex, and episode number."
        )

        st.subheader("Adjusted odds ratios and 95% confidence intervals")
        st.dataframe(
            results["odds_ratios"].round(4),
            use_container_width=True,
        )

        st.subheader("Forest plot")
        st.pyplot(
            results["figures"]["forest_plot"],
            clear_figure=False,
        )

        with st.expander("Full logistic regression summary"):
            st.text(results["model"].summary().as_text())

        metrics = results["metrics"]
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Model observations",
            f"{metrics['n_observations']:,}",
        )
        col2.metric(
            "Deaths",
            f"{metrics['deaths']:,}",
        )
        col3.metric(
            "Survivors",
            f"{metrics['survivors']:,}",
        )

    # --------------------------------------------------
    # TAB 5: MODEL EVALUATION
    # --------------------------------------------------
    with tab_evaluation:
        st.header("Model evaluation")

        metrics = results["metrics"]
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "In-sample ROC-AUC",
            f"{metrics['roc_auc_in_sample']:.3f}",
        )
        col2.metric(
            "In-sample Brier score",
            f"{metrics['brier_score_in_sample']:.3f}",
        )
        col3.metric(
            "Observed mortality",
            f"{metrics['observed_mortality']:.1%}",
        )

        st.subheader("ROC curve")
        st.pyplot(
            results["figures"]["roc_curve"],
            clear_figure=False,
        )

        st.subheader("Calibration plot")
        st.pyplot(
            results["figures"]["calibration_plot"],
            clear_figure=False,
        )

        st.warning(
            "These performance metrics are calculated on the same observations "
            "used to fit the model. They are in-sample estimates and may be "
            "optimistic. They are not held-out or external validation."
        )

    # ==================================================
    # 6. EXPORT RESULTS TO EXCEL
    # ==================================================
    st.divider()
    st.header("Download analysis results")

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        tables_to_export = [
            "quality_report",
            "descriptive_summary",
            "age_summary",
            "age_tests",
            "sex_table",
            "sex_percent",
            "sex_tests",
            "episode_table",
            "episode_rates",
            "episode_tests",
            "odds_ratios",
        ]

        for table_name in tables_to_export:
            results[table_name].to_excel(
                writer,
                sheet_name=table_name[:31],
            )

        pd.Series(
            results["metrics"],
            name="value",
        ).to_frame().to_excel(
            writer,
            sheet_name="metrics",
        )

    st.download_button(
        label="Download results workbook",
        data=excel_buffer.getvalue(),
        file_name="sepsis_diagnostic_results.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )

except Exception as exc:
    st.error(f"Analysis could not be completed: {exc}")
    st.info(
        "Check that the dataset contains the required columns with the exact "
        "names shown above, and that it includes both deaths and survivors. "
        "Check the terminal for further error details."
    )
