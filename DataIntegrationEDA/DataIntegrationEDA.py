import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



file_path = '_cancer_dataset_uae.csv'

df = pd.read_csv(file_path, encoding='utf-8', delimiter=',', na_values=["N/A", "NULL" ," "])


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
    df['comorbidities'] = df['comorbidities'].fillna('none')
    df['cause_of_death'] = df['cause_of_death'].fillna('alive')


def numericalVariablesSummary():
    print(df[['age', 'weight', 'height']].describe())

def categoricalVariablesDistribution():
    for col in df.select_dtypes(include='object').columns:
        if col in ['patient_id', 'primary_physician']:
            continue

    print(f"Value counts for '{col}':")
    value_counts = df[col].value_counts()
    percentages = df[col].value_counts(normalize=True) * 100

    # Combine counts and percentages
    for category in value_counts.index:
        count = value_counts[category]
        percent = percentages[category]
        print(f"{category}: {count} ({percent:.2f}%)")

    print("-" * 40)

def correlationAnalysis():

    correlation_matrix = df[['age', 'weight', 'height']].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    avg_age_by_cancer = df.groupby('cancer_type')['age'].mean().sort_values(ascending=False)
    print("\nAverage Age by Cancer Type:")
    print(avg_age_by_cancer)

    outcome_by_treatment = df.groupby('treatment_type')['outcome'].value_counts().unstack().fillna(0)
    print("\nOutcome Distribution by Treatment Type:")
    print(outcome_by_treatment)

def insightsVisualization():
    plt.hist(df['age'], bins=20, edgecolor='black')
    plt.title('Distribution of Patient Age')
    plt.xlabel('Age')
    plt.ylabel('Number of Patients')
    plt.grid(True)
    plt.show()

    plt.hist(df['height'], bins=20, edgecolor='black')
    plt.title('Distribution of Patient Height')
    plt.xlabel('Height (cm)')
    plt.ylabel('Number of Patients')
    plt.grid(True)
    plt.show()
    
    plt.hist(df['weight'], bins=20, edgecolor='black')
    plt.title('Distribution of Patient Weight')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Number of Patients')
    plt.grid(True)
    plt.show()

    #Bar chart
    cancer_counts = df['cancer_type'].value_counts()
    cancer_counts.plot(kind='bar')
    plt.title('Cancer Type Frequency')
    plt.xlabel('Cancer Type')
    plt.ylabel('Number of Patients')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Scatter plot
    plt.scatter(df['weight'], df['height'], alpha=0.5)
    plt.title('Weight vs Height')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Height (cm)')
    plt.grid(True)
    plt.show()

    #Box plot
    sns.boxplot(x='gender', y='age', data=df)
    plt.title('Age Distribution by Gender')
    plt.show()

    #Heat map
    corr = df[['age', 'weight', 'height']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def main():
    initialExploration()
    handleNullValues()
    handleDuplicateds()
    handleStandardization()

    
    numericalVariablesSummary()
    categoricalVariablesDistribution()
    correlationAnalysis()
    insightsVisualization()
    print("")
    df.to_csv("cancerSummary.csv", index=False)

    
if __name__ == "__main__":
    main()