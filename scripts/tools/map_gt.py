import pandas as pd
from pathlib import Path


def map_gt_fields(validation_file, dictionary_file, output_file):
    """
    Maps GT Report and GT Container diagnosis names with WHO-like fields.

    Args:
        validation_file: Path to "NM_LN_validation.xlsx"
        dictionary_file: Path to "inputs/LN_Dx_dictionary_codes_20260824.xlsx"
        output_file: Output path for "NM_LN_validation_v2.xlsx"
    """
    # Read files
    validation_df = pd.read_excel(validation_file)
    dictionary_df = pd.read_excel(dictionary_file)

    # Convert target columns to object dtype to allow string assignment
    target_cols = [
        "GT Report WHO-like Subcategories",
        "GT Report WHO-like Categories",
        "GT Report WHO-like Major Sections/Lineages",
        "GT Report Diagnosis Group 4",
        "GT Container WHO-like Subcategories",
        "GT Container WHO-like Categories",
        "GT Container WHO-like Major Sections/Lineages",
        "GT Container Diagnosis Group 4",
    ]
    for col in target_cols:
        if col in validation_df.columns:
            validation_df[col] = validation_df[col].astype("object")

    # Create mapping dictionary from the dictionary file
    # Map name to WHO-like fields
    name_to_who = {}
    for _, row in dictionary_df.iterrows():
        name = row["Diagnosis"]
        name_to_who[name] = {
            "WHO-like Subcategories": row["WHO-like Subcategories"],
            "WHO-like Categories": row["WHO-like Categories"],
            "WHO-like Major Sections/Lineages": row["WHO-like Major Sections/Lineages"],
            "Diagnostic group 4": row["Diagnostic group 4"],
        }

    # Map GT Report Diagnosis Name
    for idx, row in validation_df.iterrows():
        report_diagnosis = row["GT Report Diagnosis"]
        if report_diagnosis in name_to_who:
            mapping = name_to_who[report_diagnosis]
            validation_df.at[idx, "GT Report WHO-like Subcategories"] = mapping[
                "WHO-like Subcategories"
            ]
            validation_df.at[idx, "GT Report WHO-like Categories"] = mapping[
                "WHO-like Categories"
            ]
            validation_df.at[idx, "GT Report WHO-like Major Sections/Lineages"] = (
                mapping["WHO-like Major Sections/Lineages"]
            )
            validation_df.at[idx, "GT Report Diagnosis Group 4"] = mapping[
                "Diagnostic group 4"
            ]

    # Map GT Container Diagnosis Name
    for idx, row in validation_df.iterrows():
        container_diagnosis = row["GT Container Diagnosis"]
        if container_diagnosis in name_to_who:
            mapping = name_to_who[container_diagnosis]
            validation_df.at[idx, "GT Container WHO-like Subcategories"] = mapping[
                "WHO-like Subcategories"
            ]
            validation_df.at[idx, "GT Container WHO-like Categories"] = mapping[
                "WHO-like Categories"
            ]
            validation_df.at[idx, "GT Container WHO-like Major Sections/Lineages"] = (
                mapping["WHO-like Major Sections/Lineages"]
            )
            validation_df.at[idx, "GT Container Diagnosis Group 4"] = mapping[
                "Diagnostic group 4"
            ]

    # Export to new file
    validation_df.to_excel(output_file, index=False)
    print(f"Successfully exported to {output_file}")


if __name__ == "__main__":
    # Adjust paths as needed
    validation_file = "inputs/NM_LN_validation.xlsx"
    dictionary_file = "inputs/LN_Dx_dictionary_codes_20260824.xlsx"
    output_file = "inputs/NM_LN_validation_v2.xlsx"

    map_gt_fields(validation_file, dictionary_file, output_file)
