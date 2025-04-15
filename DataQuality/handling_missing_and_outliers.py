import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




file_path = '_cancer_dataset_uae.csv'

df = pd.read_csv(file_path, encoding='utf-8', delimiter=',')


def initialExploration():
    print(df.head())
    print(df.info())
    print(df.describe())


#Check for null values within the dataset
def handleNullValues():
    print(df.isnull().sum())

#Check for duplicates and delete them
def handleDuplicateds():
    global df
    if df.duplicated().any():
        print(f"Number of duplicate rows: {df.duplicated().sum()}")
        df = df.drop_duplicates()
        print("Duplicates removed.")
    else:
        print("No duplicate rows found.")

def handleStandardization():
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
    df['treatment_start_date'] = pd.to_datetime(df['treatment_start_date'], errors='coerce')
    df['death_date'] = pd.to_datetime(df['death_date'], errors='coerce')

def missingValuesPercentagePerColumn():
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print(missing_percentage)

def handleMissingValues():
    df['comorbidities'] = df['comorbidities'].fillna('none')
    df['cause_of_death'] = df['cause_of_death'].fillna('alive')


def identifyOutliers():
    global df_filtered
    df_filtered = df.copy()

    for column in ['age', 'weight', 'height']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        print(f"{column}:".capitalize())
        print(len(outliers))

        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[column])
        plt.title("Box Plot for Outlier Detection")
        plt.show()

        df_filtered = df_filtered[(df_filtered[column] >= lower_bound) & (df_filtered[column] <= upper_bound)]

def main():
    initialExploration()
    handleNullValues()
    handleDuplicateds()
    handleStandardization()
    missingValuesPercentagePerColumn()
    handleMissingValues()
    print(f"\nLines before removing the outliers {len(df)}")
    identifyOutliers()
    print(f"Lines after removing the outliers {len(df_filtered)}")

    print("")
    df_filtered.to_csv("cancerSummary.csv", index=False)
    
if __name__ == "__main__":
    main()