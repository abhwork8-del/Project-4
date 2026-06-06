# Project 4
# Lab 13
## Predictive Maintenance Model Training and Experiment Tracking
# Objective
The objective of this lab is to build a predictive maintenance system using machine learning techniques. Synthetic industrial sensor data is generated and used to train multiple classification models. The performance of each model is evaluated and compared.
# Dataset
A synthetic dataset containing 10,000 equipment records was generated using NumPy.
## Features
Temperature
Vibration
Pressure
RPM
Equipment Age (Days)
## Target Variable
Failure (0 = No Failure, 1 = Failure)
# Methodology
## 1. Data Generation
Synthetic sensor readings were generated using statistical distributions.
## 2. Exploratory Data Analysis (EDA)
The following analyses were performed:
Summary Statistics
Missing Value Analysis
Failure Distribution Visualization
Feature Distribution Histograms
Correlation Heatmap
## 3. Data Preprocessing
Train-Test Split (80%-20%)
Feature Scaling using StandardScaler
## 4. Model Training
Three machine learning models were trained:
1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
## 5. Evaluation Metrics
Models were evaluated using:
Accuracy
Precision
Recall
F1 Score
ROC-AUC Score
## 6. Model Comparison
All models were compared based on ROC-AUC performance.
# Results
XGBoost achieved the highest performance and was selected as the best model for predictive maintenance prediction.
# Technologies Used
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
XGBoost
MLflow
# Conclusion
The predictive maintenance system successfully identified equipment likely to fail based on sensor readings. XGBoost demonstrated superior performance and was selected for deployment in the subsequent model registry workflow.

# Lab 14
# Model Registry, Versioning and Lifecycle Management
# Objective
The objective of this lab is to demonstrate machine learning model lifecycle management, including model registration, staging, production deployment, versioning, and rollback strategies.
# Dataset
The predictive maintenance dataset from Lab 13 was reused for training and evaluation.
# Workflow
## 1. Model Training
The following models were trained:
1. Logistic Regression
2. Random Forest
3. XGBoost
## 2. Model Evaluation
Each model was evaluated using ROC-AUC score.
## 3. Best Model Selection
The model with the highest ROC-AUC score was selected as the production candidate.
## 4. Model Registration
The selected model was registered under:
PredictiveMaintenance
Version:
Version 1
## 5. Model Documentation
Model metadata was added including:
Description
Validation Status
Team Information
Framework Information
## 6. Staging Environment
The registered model was promoted to the Staging environment for validation testing.
## 7. Staging Validation
Test cases containing high-risk equipment conditions were evaluated to verify model behavior.
## 8. Production Deployment
After successful validation, the model was promoted to the Production environment.
## 9. Production Inference
A prediction function was implemented to:
Accept equipment sensor values
Perform preprocessing
Generate failure predictions
Provide maintenance recommendations
## 10. Versioning
A second model version was registered to simulate future model updates.
## 11. Rollback
A rollback procedure was demonstrated by restoring Production to Version 1.
# Technologies Used
Python
Scikit-Learn
XGBoost
MLflow Concepts
Pandas
NumPy
# Key Concepts Demonstrated
Model Registry
Model Versioning
Model Metadata
Staging Environment
Production Deployment
Inference Pipeline
Rollback Strategy
# Conclusion
The lab successfully demonstrated the complete machine learning model lifecycle. A predictive maintenance model was registered, validated, deployed, versioned, and rolled back, illustrating how ML models are managed in production environments.

## Author: Aliha Batool
## Artificial intelligence (AI)
## Lab 13 & 14
