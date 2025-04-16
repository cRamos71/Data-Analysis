# Cancer Dataset - Exploratory Data Analysis (EDA)

This repository presents a comprehensive analysis of a fictional cancer dataset from the United Arab Emirates (UAE). The project is divided into three major exercises, covering dataset import and cleaning, handling of missing values and outliers, and visual data exploration (univariate, bivariate, and multivariate).

---

## Dataset Overview

- File name: `_cancer_dataset_uae.csv`
- File origin: `kaggle.com`
- Records: 10,000 patient entries
- Columns: Demographic, clinical, and treatment-related information

---

## Objectives

### 1. Data Import and Cleaning
- Loaded dataset using `pandas` with special handling for null values (`"NULL"`, `"N/A"`, empty strings).
- Checked for and removed duplicate rows.
- Standardized column names and cleaned textual data (lowercase, trimmed spaces).
- Converted relevant date columns to proper `datetime` format.

### 2. Handling Missing Values and Outliers
- Calculated the percentage of missing values per column.
- Missing values handled:
  - `comorbidities` → filled with `'none'`
  - `cause_of_death` → filled with `'alive'`
- Detected and removed outliers using the Interquartile Range (IQR) method for:
  - `age`, `weight`, `height`

### 3. Data Visualization
#### Univariate Analysis
- `countplot` for categorical variables
- `histplot`, `kdeplot`, `boxplot`, `violinplot` for numerical variables

#### Bivariate Analysis
- Categorical vs Numerical: `boxplot`, `violinplot`, `stripplot`
- Numerical vs Numerical: `scatterplot`, `regplot`, Pearson correlation coefficient

#### Multivariate Analysis (Faceting)
- Used `FacetGrid`, `catplot`, and `relplot` to split visualizations by a third variable:
  - Example: Age distribution by gender
  - Example: Weight distribution by emirate and gender

---
