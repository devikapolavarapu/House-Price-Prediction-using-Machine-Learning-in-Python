# house_price_predictor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# --- Step 1: Load and Inspect the Data ---
print("📊 Step 1: Loading and inspecting data...")
dataset = pd.read_excel("HousePricePrediction.xlsx") # Make sure the file name matches!
print("First 5 rows:")
print(dataset.head())
print("\nDataset shape:", dataset.shape)
print("\nDataset info:")
print(dataset.info())

# --- Step 2: Data Preprocessing ---
print("\n🧹 Step 2: Cleaning and preprocessing data...")
# Drop the 'Id' column
if 'Id' in dataset.columns:
    dataset.drop('Id', axis=1, inplace=True)
    print("Dropped 'Id' column.")

# Check for missing values in SalePrice and fill with mean
if dataset['SalePrice'].isnull().sum() > 0:
    dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
    print(f"Filled {dataset['SalePrice'].isnull().sum()} missing values in 'SalePrice' with mean.")

# Drop rows with any other missing values
new_dataset = dataset.dropna()
print(f"Dropped rows with missing values. New shape: {new_dataset.shape}")

# --- Step 3: Exploratory Data Analysis (EDA) ---
print("\n🔍 Step 3: Performing Exploratory Data Analysis...")
# 3A. Correlation Heatmap
numerical_data = new_dataset.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_data.corr(), cmap='coolwarm', annot=True, fmt='.2f', center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig('correlation_heatmap.png') # Saves the plot to your project folder
print("Saved 'correlation_heatmap.png'")
# plt.show() # Uncomment this if you want the plot to pop up on your screen

# 3B. Analyze Categorical Features
object_cols = new_dataset.select_dtypes(include=['object']).columns
if not object_cols.empty:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    for i, col in enumerate(object_cols):
        value_counts = new_dataset[col].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('categorical_distribution.png')
    print("Saved 'categorical_distribution.png'")
    # plt.show()
else:
    print("No categorical features found for distribution plots.")

# --- Step 4: Encode Categorical Variables ---
print("\n🔢 Step 4: Encoding categorical variables...")
# Identify categorical columns
object_cols = new_dataset.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns to encode:", object_cols)

if object_cols:
    OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
    OH_cols.index = new_dataset.index
    OH_cols.columns = OH_encoder.get_feature_names_out(object_cols)

    df_final = new_dataset.drop(object_cols, axis=1)
    df_final = pd.concat([df_final, OH_cols], axis=1)
else:
    df_final = new_dataset # If no categorical columns, use the dataset as is

print("Final dataset shape after encoding:", df_final.shape)

# --- Step 5: Split the Data ---
print("\n✂️  Step 5: Splitting data into training and testing sets...")
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)
print(f"Training set: {X_train.shape}, Validation set: {X_valid.shape}")

# --- Step 6: Train and Evaluate Models ---
print("\n🤖 Step 6: Training and evaluating models...")
models = {
    'Support Vector Machine': svm.SVR(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=0),
    'Linear Regression': LinearRegression()
}

results = {}

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_valid)
    
    mae = mean_absolute_error(Y_valid, Y_pred)
    mape = mean_absolute_percentage_error(Y_valid, Y_pred)
    results[name] = {'MAE': mae, 'MAPE': mape}
    print(f"    {name} - MAE: ${mae:,.2f}, MAPE: {mape:.4f}")

# --- Step 7: Compare Results ---
print("\n📈 Step 7: Model Comparison")
print("\nFinal Results:")
for model, metrics in results.items():
    print(f"{model:>25}: MAE = ${metrics['MAE']:>10,.2f}, MAPE = {metrics['MAPE']:>7.4f}")

# Find the best model based on MAPE
best_model_name = min(results, key=lambda x: results[x]['MAPE'])
print(f"\n🏆 Best model is '{best_model_name}' with a MAPE of {results[best_model_name]['MAPE']:.4f}")

print("\n✅ Script finished successfully!")