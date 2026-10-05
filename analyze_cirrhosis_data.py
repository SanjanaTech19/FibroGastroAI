import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Liver Cirrhosis Mayo Clinic dataset
af = pd.read_csv("cirrhosis.csv", encoding="utf-8")

# 2. Create a copy of the dataframe
new_af = af.copy()

# 3. Convert Patient Age from days to years for clinical readability
new_af["Age_Years"] = (new_af["Age"] / 365.25).round(1)

# 4. Select key clinical biomarkers and numerical features
new_af = new_af[["ID", "Age_Years", "Sex", "Stage", "Bilirubin", "Albumin", "Copper", "Platelets", "SGOT", "Prothrombin", "Status"]]

# 5. Display the first 10 patient records
print("=== First 10 Patient Records ===")
print(new_af.head(10))

# 6. Display Summary Statistics (Mean, Median/50%, Std, Min, Max) using describe()
print("\n=== Dataset Summary Statistics using describe() ===")
stats = new_af.describe().T
stats['median'] = new_af.median(numeric_only=True)  # Highlight 50% percentile as Median
print(stats[['count', 'mean', 'std', 'min', 'median', 'max']])
