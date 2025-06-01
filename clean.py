import pandas as pd

# Load CSV
df = pd.read_csv("Titanic-Dataset.csv")

# Basic checks
print(df.info())  # Data types, missing values
print(df.duplicated().sum())  # Number of duplicate rows
print(df.isnull().sum())  # Missing values per column
print(df.describe(include='all'))  # Summary statistics

# Advanced: check for whitespace-only strings
df.applymap(lambda x: isinstance(x, str) and x.strip() == "").sum()
# Check missing data %
missing_percent = df.isnull().mean() * 100
print(missing_percent)

# Check unique values per column
unique_counts = df.nunique()
print(unique_counts)

# Check data types and memory
print(df.info())

# 1. Fill missing Embarked
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 2. Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# 3. Drop or convert Cabin
df['HasCabin'] = df['Cabin'].notnull().astype(int)
df.drop(columns='Cabin', inplace=True)  # optional

# 4. Optional: convert categorical columns
df['Pclass'] = df['Pclass'].astype('category')
df['Survived'] = df['Survived'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df.to_csv("cleaned_data.csv", index=False)

