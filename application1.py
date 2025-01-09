import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Load and Explore Data
try:
    data = pd.read_csv('application_data.csv')  # Replace with your main dataset path
    print("Application data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading application data: {e}")
    exit()

try:
    previous_data = pd.read_csv('previous_application.csv')  # Replace with your previous_application dataset path
    print("Previous application data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading previous application data: {e}")
    exit()

# Basic exploration of main dataset
print(data.info())
print(data.describe())
print(data.isnull().sum())

sns.countplot(x='TARGET', data=data)  # Replace 'TARGET' with your target variable
plt.title('Class Distribution')
plt.show()

# Basic exploration of previous_application dataset
print(previous_data.info())
print(previous_data.describe())
print(previous_data.isnull().sum())

# Step 2: Merge Datasets
# Assuming 'SK_ID_CURR' is the common key
if 'SK_ID_CURR' not in data.columns or 'SK_ID_CURR' not in previous_data.columns:
    print("SK_ID_CURR not found in one or both datasets. Ensure proper column names.")
    exit()

merged_data = data.merge(previous_data, on='SK_ID_CURR', how='left')
print("Merged data shape:", merged_data.shape)

# Step 3: Data Cleaning
# Drop columns with >30% missing values
merged_data = merged_data.dropna(thresh=0.7 * merged_data.shape[0], axis=1)
print("Remaining columns after dropping:", merged_data.shape[1])

# Fill remaining NaN values with median
numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].median())

# Remove duplicates
merged_data.drop_duplicates(inplace=True)

# Encode categorical features
categorical_cols = merged_data.select_dtypes(include=['object']).columns
if not categorical_cols.empty:
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(merged_data[categorical_cols]), 
                                columns=encoder.get_feature_names_out())
    merged_data = pd.concat([merged_data.drop(columns=categorical_cols), encoded_data], axis=1)

# Step 4: Feature Selection and Engineering
# Check available columns after cleaning
print("Available columns after cleaning:", merged_data.columns)

# Dynamic feature selection to handle missing columns
required_features = ['SK_ID_CURR', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']
available_features = [col for col in required_features if col in merged_data.columns]

if len(available_features) < len(required_features):
    print("Warning: Some selected features are missing. Proceeding with available features:", available_features)

if not available_features:
    print("Error: No selected features are available in the merged dataset. Exiting.")
    exit()

X = merged_data[available_features]
if 'TARGET' not in merged_data.columns:
    print("'TARGET' column not found in the merged data.")
    exit()

y = merged_data['TARGET']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
try:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
except Exception as e:
    print(f"Error during SMOTE resampling: {e}")
    exit()

# Step 5: Model Training
# Define a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X_train_resampled, y_train_resampled)

# Feature importance
importances = pipeline.named_steps['classifier'].feature_importances_
for feature, importance in zip(available_features, importances):
    print(f"{feature}: {importance}")


# Step 6: Model Evaluation
# Predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluation metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba)}")

# Visualize confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 7: Deployment Preparation
# Save the model for deployment
joblib.dump(pipeline, 'fraud_detection_model.pkl')
print("Model saved as 'fraud_detection_model.pkl'.")

# Load the model for inference
loaded_model = joblib.load('fraud_detection_model.pkl')

# Example inference
sample_data = X_test.iloc[:1]  # Replace with real sample data
prediction = loaded_model.predict(sample_data)
print(f"Fraud Prediction for sample data: {prediction}")
