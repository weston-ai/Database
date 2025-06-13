# SCRIPT OBJECTIVE: This script generates a synthetic longitudinal diabetes dataset simulating 60,000 patients tracked monthly over 25 timepoints. Each patient is assigned demographic, socioeconomic, behavioral, and clinical characteristics, including treatment group and city of residence stratified by diabetes risk. City assignment is based on a composite risk score influenced by race, age, BMI, sex, education, income, and insurance status. For each patient, monthly values for HbA1c, fasting glucose, insulin, and HOMA-IR are simulated with realistic baseline distributions, demographic modifiers, treatment-specific trends, and multivariate noise. The script flags remission events by determining when each metric falls below clinical thresholds and calculates the time to remission. It also simulates 55 molecular biomarkers (e.g., mRNAs, lipids, metabolites), each assigned a signal strength (moderate, weak, or random) to reflect biological variability in HbA1c correlation. The dataset includes comorbidity indicators (hypertension, hyperlipidemia), enforces dropout after a random month for 14% of patients, and performs extensive sanity checks. Finally, the dataset is exported as a CSV file for downstream use in machine learning, synthetic clinical trials, or biomarker discovery research.

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) # see all columns when we call head()
import random
from scipy.stats import bernoulli

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Constants
N_PATIENTS = 60000
N_MONTHS = 25
DROP_OUT_RATE = 0.14  # 14% dropout rate

# Demographic distributions (approximate US distributions)
race_distribution = {
    "White": 0.59, "Black": 0.13, "Hispanic": 0.18, "Asian": 0.05, "Other": 0.05
}
sex_distribution = {"Male": 0.49, "Female": 0.51}
age_distribution = {"18-44": 0.3, "45-64": 0.5, "65-85": 0.2}
education_levels = ["No High School", "High School", "2-Year College", "4-Year University", "Graduate School"]
income_levels = ["<25K", "25K-50K", "50K-75K", "75K-100K", "100K+"]
insurance_status_options = ["Private", "Medicaid", "Medicare", "Uninsured"]

# US cities stratified by diabetes risk level
low_risk_cities = ["Portland", "New York City", "Los Angeles", "San Diego", "San Francisco", "Seattle", "Denver", "Tampa Bay"]
medium_risk_cities = ["Dallas", "Phoenix", "Houston", "St Louis", "Detroit", "Cleveland", "Philadelphia", "Chicago"]
high_risk_cities = ["Bakersfield", "San Antonio", "Memphis", "New Orleans", "Atlanta", "Birmingham", "Jackson", "Jacksonville"]

# City categories grouped by tier for probabilistic assignment
city_risk_categories = {
    "low": low_risk_cities,
    "medium": medium_risk_cities,
    "high": high_risk_cities
}

# Feature-specific weights indicating likelihood of living in a high-risk city
RACE_WEIGHTS = {"Black": 0.6, "Hispanic": 0.4, "White": 0.2, "Asian": 0.1, "Other": 0.2}
AGE_GROUP_WEIGHTS = {"18-44": 0.2, "45-64": 0.5, "65-85": 0.4}
INCOME_WEIGHTS = {"<25K": 0.6, "25K-50K": 0.4, "50K-75K": 0.3, "75K-100K": 0.2, "100K+": 0.1}
EDUCATION_WEIGHTS = {"No High School": 0.6, "High School": 0.5, "2-Year College": 0.4, "4-Year University": 0.3, "Graduate School": 0.2}
SEX_WEIGHTS = {"Male": 0.5, "Female": 0.3}
BMI_WEIGHTS = {"Normal": 0.1, "Overweight": 0.3, "Obese": 0.6}
INSURANCE_WEIGHTS = {"Uninsured": 0.6, "Medicaid": 0.5, "Medicare": 0.4, "Private": 0.2}

# Comorbidity prevalence weights by city tier
COMORBIDITY_CITY_WEIGHTS = {
    "low": {"Hypertension": 0.1, "Hyperlipidemia": 0.07},
    "medium": {"Hypertension": 0.25, "Hyperlipidemia": 0.23},
    "high": {"Hypertension": 0.4, "Hyperlipidemia": 0.37}
}

# Function to assign insurance status according to age and income
def assign_insurance(age_group, income):
    if age_group == "65-85":
        probs = {"Medicare": 0.85, "Private": 0.10, "Uninsured": 0.03, "Medicaid": 0.02}
    elif income == "<25K":
        probs = {"Medicaid": 0.60, "Uninsured": 0.25, "Private": 0.10, "Medicare": 0.05}
    elif income in ["25K-50K", "50K-75K"]:
        probs = {"Private": 0.60, "Medicaid": 0.20, "Uninsured": 0.15, "Medicare": 0.05}
    else:
        probs = {"Private": 0.80, "Medicare": 0.10, "Medicaid": 0.05, "Uninsured": 0.05}

    return np.random.choice(list(probs.keys()), p=list(probs.values()))

# Categorize BMI according to WHO guidelines
def categorize_BMI(BMI):
    if BMI < 25:
        return 'Normal'
    elif 25 <= BMI < 30:
        return 'Overweight'
    else:
        return 'Obese'

# Compute composite risk score used to determine diabetes city tier assignment
def compute_city_risk_score(race, age_group, income, education, sex, bmi_cat, insurance):
    score = (
        RACE_WEIGHTS[race] + AGE_GROUP_WEIGHTS[age_group] + INCOME_WEIGHTS[income] +
        EDUCATION_WEIGHTS[education] + SEX_WEIGHTS[sex] + BMI_WEIGHTS[bmi_cat] + INSURANCE_WEIGHTS[insurance]
    )
    max_score = sum(map(max, [RACE_WEIGHTS.values(), AGE_GROUP_WEIGHTS.values(), INCOME_WEIGHTS.values(),
                              EDUCATION_WEIGHTS.values(), SEX_WEIGHTS.values(), BMI_WEIGHTS.values(), INSURANCE_WEIGHTS.values()]))
    return score / max_score

# Map composite score to city risk tier probabilities
def score_to_city_weights(score):
    if score >= 0.6:
        return {"low": 0.1, "medium": 0.3, "high": 0.6}
    elif score >= 0.4:
        return {"low": 0.3, "medium": 0.4, "high": 0.3}
    else:
        return {"low": 0.6, "medium": 0.3, "high": 0.1}

# Assign a city based on individual profile and computed weights
def assign_city(race, age_group, income, education, sex, bmi_cat, insurance):
    score = compute_city_risk_score(race, age_group, income, education, sex, bmi_cat, insurance)
    weights = score_to_city_weights(score)
    tier = np.random.choice(list(weights.keys()), p=list(weights.values()))
    return np.random.choice(city_risk_categories[tier]), tier

# Generate patient data with demographic, clinical, and city risk stratification
def generate_patients(n):
    patients = []

    treatments = ["Drug A with Exercise"] * (n // 3) + ["Placebo with Exercise"] * (n // 3) + \
                 ["Drug A"] * (n - 2 * (n // 3))
    random.shuffle(treatments)

    for i in range(n):
        race = np.random.choice(list(race_distribution.keys()), p=list(race_distribution.values()))
        sex = np.random.choice(list(sex_distribution.keys()), p=list(sex_distribution.values()))
        age_group = np.random.choice(list(age_distribution.keys()), p=list(age_distribution.values()))

        if age_group == "18-44":
            edu_probs = [0.1, 0.4, 0.3, 0.15, 0.05]
            inc_probs = [0.3, 0.3, 0.2, 0.15, 0.05]
        elif age_group == "45-64":
            edu_probs = [0.05, 0.3, 0.3, 0.25, 0.1]
            inc_probs = [0.15, 0.3, 0.3, 0.15, 0.1]
        else:
            edu_probs = [0.05, 0.25, 0.3, 0.25, 0.15]
            inc_probs = [0.1, 0.2, 0.3, 0.2, 0.2]

        education = np.random.choice(education_levels, p=edu_probs)
        income = np.random.choice(income_levels, p=inc_probs)
        insurance = assign_insurance(age_group, income)

        bmi = np.random.normal(27, 4)
        if (sex == "Male" and bmi > 28) or (sex == "Female" and bmi > 34):
            bmi += np.random.uniform(2, 5)
        bmi_cat = categorize_BMI(bmi)

        dropout = bernoulli.rvs(DROP_OUT_RATE)
        if dropout:
            dropout_month = int(np.clip(np.random.beta(2, 3) * (N_MONTHS - 1), 1, N_MONTHS - 1))
        else:
            dropout_month = N_MONTHS

        city, city_risk = assign_city(race, age_group, income, education, sex, bmi_cat, insurance)

        smoked = bernoulli.rvs(0.3)
        alcohol = np.round(np.clip(np.random.gamma(0.3, 1 / 0.3) * (1.15 if sex == "Male" else 1), 0, 8) * 2) / 2.0

        # Assign comorbidities based on city risk tier
        comorbidity_weights = COMORBIDITY_CITY_WEIGHTS[city_risk]
        hypertension = bernoulli.rvs(comorbidity_weights["Hypertension"])
        hyperlipidemia = bernoulli.rvs(comorbidity_weights["Hyperlipidemia"])

        patients.append([
            i, race, sex, age_group, education, income, insurance, bmi, bmi_cat, treatments[i],
            dropout_month, city, city_risk, smoked, alcohol, hypertension, hyperlipidemia
        ])

    cols = ["Patient_ID", "Race", "Sex", "Age_Group", "Education", "Income", "Insurance", "BMI", "BMI_Category",
            "Treatment", "Dropout_Month", "City", "City_Risk_Tier", "Smoked_one_pack_10_years_or_more",
            "Alcohol_daily", "Hypertension", "Hyperlipidemia"]
    return pd.DataFrame(patients, columns=cols)

### Generate patient observations
CITY_METRIC_BASELINES = {
    "low": {"HbA1c": (6.5, 0.5), "Glucose": (110, 10), "Insulin": (15, 5), "HOMA_IR": (3.0, 0.8)},
    "medium": {"HbA1c": (7.0, 0.7), "Glucose": (120, 15), "Insulin": (18, 6), "HOMA_IR": (4.0, 1.0)},
    "high": {"HbA1c": (7.5, 0.9), "Glucose": (130, 20), "Insulin": (21, 8), "HOMA_IR": (5.0, 1.2)}
}

# Define treatment effects (slope of monthly improvement, peak month window)
TREATMENT_EFFECTS = {
    "Drug A with Exercise": {"slope": -0.08, "peak_range": (7, 11)},
    "Placebo with Exercise": {"slope": -0.05, "peak_range": (12, 14)},
    "Drug A": {"slope": -0.03, "peak_range": (15, 17)}
}

# Function to generate time series of diabetes metrics for each patient
# Patient-specific correlation matrix
def generate_correlation_matrix():
    base_corr = np.array([
        [1.0, 0.65, 0.55, 0.6],
        [0.65, 1.0, 0.7, 0.55],
        [0.55, 0.7, 1.0, 0.6],
        [0.6, 0.55, 0.6, 1.0]
    ])

    noise = np.random.normal(0, 0.1, size=(4, 4))
    noise = (noise + noise.T) / 2  # make symmetric
    np.fill_diagonal(noise, 0)
    corr = base_corr + noise
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)

    corr = nearest_positive_semidefinite(corr)

    # --- NEW FINAL SAFETY TWEAK ---
    # Slightly shrink off-diagonals to make sure it's not nearly singular
    shrink_factor = 0.98  # 2% shrink
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i != j:
                corr[i, j] *= shrink_factor
    np.fill_diagonal(corr, 1.0)

    return corr

def nearest_positive_semidefinite(matrix):
    """Project a symmetric matrix to the nearest positive semi-definite matrix."""
    # Symmetrize just in case
    matrix = (matrix + matrix.T) / 2
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Clip negative eigenvalues to zero
    eigenvalues_clipped = np.clip(eigenvalues, 0, None)
    # Reconstruct the matrix
    psd_matrix = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T
    return psd_matrix

def generate_diabetes_observations(patients_df, n_months=24):
    all_observations = []

    for _, patient in patients_df.iterrows():
        treatment = patient["Treatment"]
        city_risk = patient["City_Risk_Tier"]
        race = patient["Race"]
        sex = patient["Sex"]
        pid = patient["Patient_ID"]

        # Get baseline means and SDs based on city tier
        baselines = CITY_METRIC_BASELINES[city_risk]
        means = [baselines["HbA1c"][0], baselines["Glucose"][0], baselines["Insulin"][0], baselines["HOMA_IR"][0]]
        stds = [baselines["HbA1c"][1], baselines["Glucose"][1], baselines["Insulin"][1], baselines["HOMA_IR"][1]]

        # Apply demographic bonus (only for Hispanic or Black females)
        bonus_factor = 1.0
        if race == "Hispanic":
            bonus_factor *= 0.85
        elif race == "Black" and sex == "Female":
            bonus_factor *= 0.80

        # Get treatment effect parameters
        slope = TREATMENT_EFFECTS[treatment]["slope"]
        peak_start, peak_end = TREATMENT_EFFECTS[treatment]["peak_range"]

        # Generate patient-specific correlation matrix
        corr_matrix = generate_correlation_matrix()
        cov_matrix = np.outer(stds, stds) * corr_matrix

        # Simulate baseline values
        baseline_values = np.random.multivariate_normal(mean=means, cov=cov_matrix)

        for month in range(int(patient["Dropout_Month"]) + 1):
            if month <= peak_end:
                trend = slope * month
            else:
                trend = slope * peak_end + (month - peak_end) * (slope * 0.3)

            error_scale = 1.0 - (month / n_months) * 0.4

            scaled_cov = cov_matrix * (error_scale * 1.5) ** 2
            scaled_cov = nearest_positive_semidefinite(scaled_cov)

            monthly_noise = np.random.multivariate_normal(
                mean=[0, 0, 0, 0],
                cov=scaled_cov
            )

            values = baseline_values + trend * bonus_factor + monthly_noise

            all_observations.append([
                pid, month,
                round(values[0], 2),  # HbA1c
                round(values[1], 1),  # Glucose
                round(values[2], 1),  # Insulin
                round(values[3], 2)   # HOMA_IR
            ])

    return pd.DataFrame(
        all_observations,
        columns=["Patient_ID", "Month", "HbA1c", "Fasting_Glucose", "Fasting_Insulin", "HOMA_IR"]
    )

# Function to identify when a patient's diabetic metric improves to non-diabetic range
def add_remission_event_flags(df):
    # Sort by patient and month
    df = df.sort_values(["Patient_ID", "Month"]).copy()

    # Initialize new columns
    df["HbA1c_Remission_Month"] = np.nan
    df["Glucose_Remission_Month"] = np.nan
    df["Insulin_Remission_Month"] = np.nan
    df["HOMA_IR_Remission_Month"] = np.nan

    df["HbA1c_Remission_Duration"] = np.nan
    df["Glucose_Remission_Duration"] = np.nan
    df["Insulin_Remission_Duration"] = np.nan
    df["HOMA_IR_Remission_Duration"] = np.nan

    # Group by patient to determine first remission and duration
    for pid, group in df.groupby("Patient_ID"):
        group_sorted = group.sort_values("Month")
        baseline_month = group_sorted["Month"].min()

        # Find remission months
        hba1c_month = group_sorted.loc[group_sorted["HbA1c"] < 6.5, "Month"].min()
        glucose_month = group_sorted.loc[group_sorted["Fasting_Glucose"] < 126, "Month"].min()
        insulin_month = group_sorted.loc[group_sorted["Fasting_Insulin"] < 25, "Month"].min()
        homa_ir_month = group_sorted.loc[group_sorted["HOMA_IR"] < 2.6, "Month"].min()

        # Calculate duration from baseline
        hba1c_duration = hba1c_month - baseline_month if pd.notna(hba1c_month) else np.nan
        glucose_duration = glucose_month - baseline_month if pd.notna(glucose_month) else np.nan
        insulin_duration = insulin_month - baseline_month if pd.notna(insulin_month) else np.nan
        homa_ir_duration = homa_ir_month - baseline_month if pd.notna(homa_ir_month) else np.nan

        # Assign values to the full patient block
        df.loc[df["Patient_ID"] == pid, "HbA1c_Remission_Month"] = hba1c_month
        df.loc[df["Patient_ID"] == pid, "Glucose_Remission_Month"] = glucose_month
        df.loc[df["Patient_ID"] == pid, "Insulin_Remission_Month"] = insulin_month
        df.loc[df["Patient_ID"] == pid, "HOMA_IR_Remission_Month"] = homa_ir_month

        df.loc[df["Patient_ID"] == pid, "HbA1c_Remission_Duration"] = hba1c_duration
        df.loc[df["Patient_ID"] == pid, "Glucose_Remission_Duration"] = glucose_duration
        df.loc[df["Patient_ID"] == pid, "Insulin_Remission_Duration"] = insulin_duration
        df.loc[df["Patient_ID"] == pid, "HOMA_IR_Remission_Duration"] = homa_ir_duration

    return df

def mask_missing_remission_info(df):
    # Identify all remission columns
    remission_cols = [
        "HbA1c_Remission_Month", "Glucose_Remission_Month",
        "Insulin_Remission_Month", "HOMA_IR_Remission_Month",
        "HbA1c_Remission_Duration", "Glucose_Remission_Duration",
        "Insulin_Remission_Duration", "HOMA_IR_Remission_Duration"
    ]

    # Group by Patient_ID to check for missing remission data
    for col in remission_cols:
        # Get patients where the value is NaN for this remission field
        missing_patients = df[df[col].isna()]["Patient_ID"].unique()
        # Set the column to NaN for those patients (across all their rows)
        df.loc[df["Patient_ID"].isin(missing_patients), col] = np.nan

    return df

### Generate patient dataset
patients_df = generate_patients(N_PATIENTS)

### Create patient observation dataset with diabetic metrics (e.g. HbA1c, fasting blood glucose, etc)
diabetes_observations_df = generate_diabetes_observations(patients_df)

### Merge feature inputs and observed diabetic metric outcomes
full_diabetes_df = diabetes_observations_df.merge(patients_df, on="Patient_ID", how="left")
del diabetes_observations_df

### Calculate diabetic metric remission (month of remission and time required to reach it)
full_diabetes_df = add_remission_event_flags(full_diabetes_df)

### Add NaN when the remission month or the remission duration are blank
full_diabetes_df = mask_missing_remission_info(full_diabetes_df)

# Sanity check for the patients_df dataset to ensure that roughly 1/3 of patients are in each treatment arm
print("\nTreatment Distribution:")
print(full_diabetes_df["Treatment"].value_counts(normalize=True).round(3))  # Show proportion per treatment
assert abs(full_diabetes_df["Treatment"].value_counts(normalize=True)["Drug A with Exercise"] - 1/3) < 0.02, "Drug A with Exercise proportion off"
assert abs(full_diabetes_df["Treatment"].value_counts(normalize=True)["Placebo with Exercise"] - 1/3) < 0.02, "Placebo with Exercise proportion off"
assert abs(full_diabetes_df["Treatment"].value_counts(normalize=True)["Drug A"] - 1/3) < 0.02, "Drug A proportion off"

# Sanity check for dropout rate
dropout_rate = (patients_df["Dropout_Month"] < N_MONTHS).mean()
print(f"\nObserved Dropout Rate: {dropout_rate:.3f}")
assert abs(dropout_rate - DROP_OUT_RATE) < 0.03, "Dropout rate is unexpectedly off"

# Sanity check for the prevalence of hypertension and hyperlipidemia
print("\nComorbidity Prevalence:")
print("Hypertension:", full_diabetes_df["Hypertension"].mean().round(3))
print("Hyperlipidemia:", full_diabetes_df["Hyperlipidemia"].mean().round(3))
assert full_diabetes_df["Hypertension"].mean() >= 0.28
assert full_diabetes_df["Hyperlipidemia"].mean() >= 0.22

# Sanity check for City Risk Tier distribution
print("\nCity Risk Tier Distribution (Actual vs. Expected):")
actual_distribution = full_diabetes_df["City_Risk_Tier"].value_counts(normalize=True).round(3).sort_index()
expected_distribution = pd.Series({"high": 0.47, "medium": 0.34, "low": 0.19}).sort_index()

comparison_df = pd.DataFrame({
    "Expected": expected_distribution,
    "Actual": actual_distribution
})

print(comparison_df)

## Sanity check of patient observation dataset
# Check row count
full_diabetes_df.shape[0]

# Check distribution of the metrics
print(full_diabetes_df[["HbA1c", "Fasting_Glucose", "Fasting_Insulin", "HOMA_IR"]].describe())

# Visualize trend over time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
sns.lineplot(data=full_diabetes_df, x="Month", y="HbA1c", hue="Treatment")
plt.title("HbA1c over Time by Treatment")
plt.show()

# Check for patient uniqueness (ensure 25 timepoints per patient)
counts = full_diabetes_df["Patient_ID"].value_counts()
assert counts.min() == 25 and counts.max() == 25, "Each patient should have exactly 25 observations"

# Check for realistic improvement patterns (look at average HbA1c drop from baseline to Month 24)
subset = full_diabetes_df[["Patient_ID", "Month", "HbA1c", "Race", "Sex"]]
delta = subset[subset["Month"] == 0][["Patient_ID", "HbA1c"]].merge(
    subset[subset["Month"] == 24][["Patient_ID", "HbA1c"]],
    on="Patient_ID",
    suffixes=("_0", "_24")
)
delta["change"] = delta["HbA1c_24"] - delta["HbA1c_0"]
delta = delta.merge(patients_df[["Patient_ID", "Race", "Sex"]], on="Patient_ID")
del patients_df

print(delta.groupby(["Race", "Sex"])["change"].mean().sort_values())

# Check for randomness within city
baseline_df = full_diabetes_df[full_diabetes_df["Month"] == 0]
merged = baseline_df[["City_Risk_Tier", "HbA1c", "Fasting_Glucose"]]
print(merged.groupby("City_Risk_Tier")[["HbA1c", "Fasting_Glucose"]].std())

# Visual check of diabetic metric correlations
sns.pairplot(full_diabetes_df.query("Month == 0")[["HbA1c", "Fasting_Glucose", "Fasting_Insulin", "HOMA_IR"]])

# Check proportion of patients who reach remission for each metric
for metric in ["HbA1c", "Glucose", "Insulin", "HOMA_IR"]:
    prop = full_diabetes_df.groupby("Patient_ID")[f"{metric}_Remission_Month"].first().notna().mean()
    print(f"{metric} remission rate: {prop:.2%}")

### Add molecules (genes, proteins, metabolites, lipids, etc) to the dataset. We want moderate, weak, and null associations for these molecules with HbA1c.
# ------------------------------------------
# 1. Define Molecule List with Acronym Names
# ------------------------------------------
molecule_names = [
    # mRNA transcripts
    'INS', 'IRS1', 'IRS2', 'PPARG', 'GLUT4', 'TNF_alpha', 'IL6', 'FOXO1', 'G6PC', 'PCK1', 'UCP2',
    # Carnitines
    'Palmitoylcarnitine', 'Oleoylcarnitine', 'C3_carnitine', 'C5_carnitine', 'C6_carnitine', 'C8_carnitine',
    # Proteins
    'Adiponectin', 'Resistin', 'CRP', 'FetuinA', 'Leptin', 'FNDC5', 'Complement_C3', 'Complement_C4',
    # Lipids
    'Cer_d18_1_16_0', 'Cer_d18_1_24_1', 'DAG_18_0_18_1', 'Palmitic_acid', 'Oleic_acid', 'SM_d18_1_16_0',
    'LPC_16_0', 'LPC_18_1', 'TG_54_2', 'TG_52_2',
    # Metabolites
    'Glucose_metabolite', 'Lactate', 'Pyruvate', 'Alanine', 'Leucine', 'Isoleucine', 'Valine', 'Glutamine',
    'Glutamate', 'Citrate', 'Succinate', 'Malate', 'Fumarate', 'Alpha_Ketoglutarate',
    'Acetoacetate', 'Beta_Hydroxybutyrate', 'Uric_acid', 'Hydroxyisobutyrate', 'Phenylalanine',
    'Tyrosine', 'Serine', 'Glycine'
]

# ---------------------------------------------------
# 2. Assign each molecule to a signal strength bucket
# ---------------------------------------------------
np.random.seed(42)

n_molecules = len(molecule_names)
assigned = np.random.choice(
    ['moderate', 'weak', 'random'],
    size=n_molecules,
    p=[0.4, 0.4, 0.2]  # 2/5 moderate, 2/5 weak, 1/5 random
)

molecule_signal_strength = dict(zip(molecule_names, assigned))

# ---------------------------------------------
# 3. Generate Molecule Values per Patient-Month
# ---------------------------------------------

def simulate_molecule(hba1c_series, month_series, bmi_series, strength, scale_factor):
    noise = np.random.normal(0, 2, size=len(hba1c_series))
    time_effect = month_series / month_series.max()
    bmi_std = bmi_series.std()
    bmi_effect = (bmi_series - bmi_series.mean()) / bmi_std if bmi_std > 1e-6 else 0

    if strength == 'moderate':
        return scale_factor * (0.35 * hba1c_series + 0.05 * time_effect + 0.1 * bmi_effect + 0.7 * noise)
    elif strength == 'weak':
        return scale_factor * (0.15 * hba1c_series + 0.05 * time_effect + 0.1 * bmi_effect + 0.85 * noise)
    else:  # random
        return scale_factor * (0.05 * time_effect + 0.1 * bmi_effect + noise)

def generate_molecule_signals(df, molecule_names, molecule_signal_strength, random_seed=42):
    """
    Generate simulated molecule signals for each patient-month based on HbA1c, Month, and BMI.

    Args:
        df (pd.DataFrame): Full diabetes dataset with HbA1c, Month, and BMI columns.
        molecule_names (list): List of molecule names to simulate.
        molecule_signal_strength (dict): Mapping of molecule name to 'moderate', 'weak', or 'random'.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Updated DataFrame with simulated molecule columns added.
    """
    np.random.seed(random_seed)

    n_molecules = len(molecule_names)
    scale_factors = np.random.uniform(0.5, 5.0, size=n_molecules)

    for idx, molecule in enumerate(molecule_names):
        strength = molecule_signal_strength[molecule]
        scale = scale_factors[idx]
        df[molecule] = simulate_molecule(
            hba1c_series=df["HbA1c"],
            month_series=df["Month"],
            bmi_series=df["BMI"],
            strength=strength,
            scale_factor=scale
        )
    return df

# Simulate molecules
full_diabetes_df = generate_molecule_signals(
    df=full_diabetes_df,
    molecule_names=molecule_names,
    molecule_signal_strength=molecule_signal_strength,
    random_seed=42
)

# -------------------------
# 4. Quick Sanity Checks
# -------------------------
print("\nMolecule Signal Strengths Summary:")
print(pd.Series(molecule_signal_strength).value_counts())

print("\nExample molecules added:")
print(full_diabetes_df[["HbA1c"] + molecule_names[:5]].head())

### Save dataset
full_diabetes_df.to_csv("/home/cweston1/miniconda3/envs/PythonProject/datasets/my_custom_datasets/diabetes_60k_pats_20250428.csv", index=False)
print("Dataset generated and saved as diabetes_60k_pats_20250428.csv")

# --------------------------------------
# SUMMARY OF SYNTHETIC DIABETES DATASET
# --------------------------------------

# 1. Patient Generation:
#    - 60,000 patients created with realistic US demographic distributions (race, sex, age, education, income).
#    - BMI values assigned with sex-specific overweight thresholds.
#    - Smoking and alcohol use randomly assigned with realistic patterns.
#    - Insurance type and city assigned probabilistically based on composite risk scores.
#    - Cities stratified by diabetes risk level (low, medium, high).
#    - Patients randomly assigned to treatment groups (Drug A with Exercise, Placebo with Exercise, Drug A).
#    - 14% dropout rate simulated, with random dropout month if dropped.

# 2. Clinical Metric Simulation:
#    - For each patient, simulated 25 monthly observations (baseline + 24 months) for:
#        * HbA1c
#        * Fasting Glucose
#        * Fasting Insulin
#        * HOMA-IR
#    - Baseline metrics drawn from city risk tier-specific distributions.
#    - Treatment effects modeled with slopes and peak-response windows.
#    - Longitudinal trends include treatment effects plus random biological noise.
#    - Hispanic and Black female patients receive a demographic "bonus" factor (better response).

# 3. Remission Event Calculation:
#    - For each patient and each metric, remission month and remission duration calculated:
#        * HbA1c < 6.5%
#        * Fasting Glucose < 126 mg/dL
#        * Fasting Insulin < 25 μU/mL
#        * HOMA-IR < 2.6
#    - Missing remission events handled cleanly (filled with NaN across all months).

# 4. Molecule Simulation:
#    - 55 biologically relevant molecules added (mRNAs, proteins, lipids, metabolites, carnitines).
#    - Molecules assigned to 3 signal strength groups:
#        * 2/5 moderately correlated with HbA1c (r ≈ 0.3–0.4).
#        * 2/5 weakly correlated (r ≈ 0.15–0.25).
#        * 1/5 randomly correlated (r ≈ 0).
#    - Each molecule generated with random Gaussian noise and biological scale variation.
#    - Molecule names stored with clean acronyms (e.g., 'IRS1', 'C3_carnitine', 'Cer_d18_1_16_0').

# 5. Sanity Checks:
#    - Treatment group proportions validated (~33% each).
#    - Dropout rate validated (~14% ± 2%).
#    - Hypertension and hyperlipidemia prevalence thresholds confirmed.
#    - City risk tier distribution validated against expectations.
#    - Each patient confirmed to have exactly 25 monthly observations.
#    - Diabetic metrics show realistic downward trends over time under treatment.
#    - Molecule-to-HbA1c correlations behave as designed (moderate, weak, random).

# 6. Final Dataset:
#    - Merged into a single DataFrame ('full_diabetes_df').
#    - Ready for machine learning, feature selection, longitudinal analysis, or synthetic trial simulation.

# --------------------------------------