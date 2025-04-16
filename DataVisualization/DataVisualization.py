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



def countplots(columns):
    for col in columns:
        if col in df.columns:
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x=col, order=df[col].value_counts().index)
            plt.title(f'Distribution of {col.replace("_", " ").title()}')
            plt.xlabel(col.replace("_", " ").title())
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

def numerical_univariate(columns):
    for col in columns:
        if col in df.columns:
            # Histogram
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=False, bins=30)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # KDE Plot
            plt.figure(figsize=(8, 4))
            sns.kdeplot(df[col], fill=True)
            plt.title(f'Density Curve (KDE) of {col}')
            plt.xlabel(col)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Boxplot
            plt.figure(figsize=(8, 2))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Violin Plot
            plt.figure(figsize=(8, 4))
            sns.violinplot(x=df[col])
            plt.title(f'Violin Plot of {col}')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def bivariate_categorical_vs_numerical(cat_col, num_col):
    if cat_col in df.columns and num_col in df.columns:
        # Boxplot
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f"Boxplot of {num_col} by {cat_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Violin plot
        plt.figure(figsize=(10, 5))
        sns.violinplot(x=cat_col, y=num_col, data=df)
        plt.title(f"Violin plot of {num_col} by {cat_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Stripplot
        plt.figure(figsize=(10, 5))
        sns.stripplot(x=cat_col, y=num_col, data=df, jitter=True, alpha=0.5)
        plt.title(f"Stripplot of {num_col} by {cat_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def bivariate_numerical_vs_numerical(x_col, y_col, hue_col=None):
    if x_col in df.columns and y_col in df.columns:
        # Scatterplot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
        plt.title(f"Scatterplot: {x_col} vs {y_col}" + (f" by {hue_col}" if hue_col else ""))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Regression line
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={'alpha': 0.3})
        plt.title(f"Regression Line: {x_col} vs {y_col}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Correlation coefficient
        correlation = df[[x_col, y_col]].corr().iloc[0, 1]
        print(f"Correlation coefficient between {x_col} and {y_col}: {correlation:.2f}")

# Histogram using FacetGrid
def facet_histogram(col, facet_by):
    g = sns.FacetGrid(df, col=facet_by, height=4, aspect=1.2)
    g.map(sns.histplot, col, bins=20)
    g.set_axis_labels(col, "Count")
    g.fig.suptitle(f"Histogram of {col.title()} by {facet_by.title()}", y=1.05)
    plt.show()

# Boxplot by two categories
def facet_boxplot(x, y, hue=None):
    sns.catplot(data=df, x=x, y=y, hue=hue, kind='box', height=5, aspect=1.5)
    plt.title(f"Boxplot of {y.title()} by {x.title()}" + (f" and {hue.title()}" if hue else ""))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Scatterplot, faceted by Region (Emirate)
def facet_scatterplot(x, y, col_by):
    sns.relplot(data=df, x=x, y=y, col=col_by, kind='scatter', height=4, aspect=1)
    plt.suptitle(f"{y.title()} vs {x.title()} by {col_by.title()}", y=1.03)
    plt.tight_layout()
    plt.show()

def main():
    initialExploration()
    handleNullValues()
    handleDuplicateds()
    handleStandardization()


    #Univariate Visualizations
    #Categorical:
    categorical_columns = ['gender', 'nationality', 'cancer_type', 'outcome', 'treatment_type']
    countplots(categorical_columns)
    #Numerical:
    numerical_univariate(['age', 'weight', 'height'])


    #Bivariate Visualizations
    #CategoricalVsNumerical
    bivariate_categorical_vs_numerical("gender", "age")
    #NumericalVsNumerical
    bivariate_numerical_vs_numerical("weight", "height", hue_col="gender")


    #Multivariate  Visualizations
    facet_histogram("age", "gender")
    facet_boxplot("emirate", "weight", hue="gender")
    facet_scatterplot("age", "height", col_by="emirate")

if __name__ == "__main__":
    main()