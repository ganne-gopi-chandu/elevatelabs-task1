import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the cleaned data
df = pd.read_csv("cleaned_data.csv")

# 2. Drop unnecessary columns
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# 3. Fill missing values (before encoding or outlier removal)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# 4. Visualize boxplots BEFORE outlier removal
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x=df['Age'], ax=axes[0, 0])
axes[0, 0].set_title("Age (Before Outlier Removal)")
sns.boxplot(x=df['Fare'], ax=axes[0, 1])
axes[0, 1].set_title("Fare (Before Outlier Removal)")

# 5. Remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

df = remove_outliers(df, 'Age')
df = remove_outliers(df, 'Fare')
df.loc[df['Age'] > 100, 'Age'] = df['Age'].median()
print("Max age:", df['Age'].max())

# 6. Visualize boxplots AFTER outlier removal
sns.boxplot(x=df['Age'], ax=axes[1, 0])
axes[1, 0].set_title("Age (After Outlier Removal)")
sns.boxplot(x=df['Fare'], ax=axes[1, 1])
axes[1, 1].set_title("Fare (After Outlier Removal)")
plt.tight_layout()
plt.show()

# 7. Encode categorical variables
label_enc = LabelEncoder()
df['Sex'] = label_enc.fit_transform(df['Sex'])  # male=1, female=0
df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=True)

# 8. Separate features and target
X = df.drop(columns=['Survived'])
y = df['Survived'].astype(int)

# 9. Scale numerical features
scaler = StandardScaler()
X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

# 10. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 11. Save preprocessed data
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
df.to_csv("cleaned_data_final.csv", index=False)

# 12. Final output
print("✅ Data preprocessing complete.")
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

