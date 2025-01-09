# Credit-card-fraud-detection
Project Overview
This project utilizes real-world transaction data to train machine learning models that can effectively classify transactions as fraudulent or non-fraudulent. By using features like credit amount, income, age, and past transaction history, the model is trained to distinguish fraudulent activities from legitimate ones.

Key tasks and steps involved:

Data Collection & Preprocessing
Feature Engineering
Model Training
Model Evaluation
Performance Analysis
Data Description
The dataset consists of two main sources:

Application Data: Contains customer information such as income, gender, car ownership, and other demographic features.
Previous Application Data: Contains transaction-related information, such as loan amounts, credit ratings, and other financial details.
The merged dataset contains:

307,511 customer records with 122 features in the application dataset.
1.67 million transaction records with 37 features in the previous application dataset.
After cleaning and preprocessing, the merged dataset has 1,430,155 records and 158 features, with columns like SK_ID_CURR, TARGET (fraud indicator), AMT_CREDIT, AMT_INCOME_TOTAL, and many others.

Modeling and Machine Learning
The project applies a classification algorithm to predict whether a transaction is fraudulent or not (binary classification). The steps include:

Data Cleaning:

Handling missing values.
Merging relevant datasets.
Dropping unnecessary or redundant features.
Feature Selection:

Selecting the most impactful features for fraud detection.
Features such as AMT_INCOME_TOTAL, AMT_CREDIT, and SK_ID_CURR are crucial for the model.
Model Training:

Several machine learning models (e.g., logistic regression, random forest) are trained on the data.
The model is tuned to achieve the best balance between precision and recall.
Model Evaluation:

Accuracy: 98% of transactions are classified correctly.
Precision: 83% of predicted fraud cases are truly fraudulent.
Recall: 95% of actual frauds are detected.
ROC AUC Score: 0.98, indicating the model's ability to distinguish between fraudulent and non-fraudulent transactions.
Key Results
Confusion Matrix:

256,396 legitimate transactions were correctly classified.
23,618 fraudulent transactions were correctly identified.
A few fraudulent transactions were misclassified as legitimate, but overall performance is strong.
Classification Report:

Precision and recall scores are high, showing the model's effectiveness in detecting fraud.
The ROC AUC score of 0.98 indicates excellent performance.
Technologies Used
Python: Main programming language for data manipulation, model training, and evaluation.
Pandas: Data processing and manipulation.
Scikit-learn: Machine learning library for training models and evaluating performance.
Matplotlib & Seaborn: Data visualization to display results and insights.
Jupyter Notebooks: For experimentation and analysis.
How to Run the Project
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install the necessary dependencies:

bash
Copy code
pip install -r requirements.txt
Run the main Python script:

bash
Copy code
python application1.py
The model will output the evaluation metrics, confusion matrix, and ROC AUC score, along with insights into the fraud detection performance.
Conclusion
This project demonstrates the application of machine learning for fraud detection in financial transactions. The model achieves high accuracy, precision, and recall, making it effective for identifying potential fraudulent activities in a large dataset. This solution can be expanded with more advanced models, feature engineering, and real-time data processing.
