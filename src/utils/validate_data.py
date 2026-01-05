import great_expectations as ge
from typing import Tuple, List
import pandas as pd


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Data validation for Telco Customer Churn dataset using Great Expectations.

    This validation runs on RAW data (before preprocessing) and focuses on:
    - Schema correctness
    - Business logic constraints
    - Safe numeric validation where applicable
    """

    print("🔍 Starting data validation with Great Expectations...")

    # ------------------------------------------------------------------
    # IMPORTANT: Handle Telco dataset quirks safely
    # TotalCharges comes as string with blank values -> convert for validation
    # ------------------------------------------------------------------
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"]
            .replace(" ", None)
            .astype(float)
        )

    # Create validator (STABLE API)
    validator = ge.from_pandas(df)

    # =========================
    # SCHEMA VALIDATION
    # =========================
    print("   📋 Validating schema...")

    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    for col in required_columns:
        validator.expect_column_to_exist(col)

    validator.expect_column_values_to_not_be_null("customerID")

    # =========================
    # BUSINESS LOGIC CHECKS
    # =========================
    print("   💼 Validating business rules...")

    validator.expect_column_values_to_be_in_set("gender", ["Male", "Female"])

    yes_no_cols = ["Partner", "Dependents", "PhoneService", "Churn"]
    for col in yes_no_cols:
        validator.expect_column_values_to_be_in_set(col, ["Yes", "No"])

    validator.expect_column_values_to_be_in_set(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    validator.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )

    # =========================
    # NUMERIC VALIDATION
    # =========================
    print("   📊 Validating numeric ranges...")

    validator.expect_column_values_to_be_between(
        "tenure",
        min_value=0,
        max_value=120
    )

    validator.expect_column_values_to_be_between(
        "MonthlyCharges",
        min_value=0,
        max_value=200
    )

    validator.expect_column_values_to_be_between(
        "TotalCharges",
        min_value=0,
        mostly=0.95  # allow small % of edge cases
    )

    validator.expect_column_values_to_not_be_null("tenure")
    validator.expect_column_values_to_not_be_null("MonthlyCharges")

    # =========================
    # CONSISTENCY CHECK
    # =========================
    print("   🔗 Validating consistency...")

    validator.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95
    )

    # =========================
    # RUN VALIDATION
    # =========================
    print("   ⚙️  Running validation suite...")
    results = validator.validate()

    # =========================
    # PROCESS RESULTS
    # =========================
    failed_expectations = [
        r["expectation_config"]["expectation_type"]
        for r in results["results"]
        if not r["success"]
    ]

    total_checks = len(results["results"])
    passed_checks = total_checks - len(failed_expectations)

    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {len(failed_expectations)}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return results["success"], failed_expectations
