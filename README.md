# Project 4
# Lab 13
## Predictive Maintenance using Machine Learning and MLflow
## Objective:
The objective of this lab is to build a predictive maintenance system for industrial equipment using machine learning models and track experiments using MLflow.
## Tools and Technologies:
- Python
- VS Code
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- MLflow
## Dataset:
A synthetic predictive maintenance dataset was generated containing 10,000 samples with the following features:
1. Temperature
2. Vibration
3. Pressure
4. RPM
5. Age_Days
## Target Variable:
- Failure (0 = No Failure, 1 = Failure)
## Methodology:
1. Generated synthetic maintenance dataset.
2. Performed exploratory data analysis (EDA).
3. Checked for missing values.
4. Visualized failure distribution.
5. Split dataset into training and testing sets.
6. Applied feature scaling using StandardScaler.
7. Trained three machine learning models:
   - Logistic Regression
   - Random Forest
   - XGBoost
8. Evaluated models using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC Score
9. Logged experiments, parameters, metrics, and models using MLflow.
10. Compared model performance using MLflow UI.
## Results:
- Logistic Regression
- Random Forest
- XGBoost
The best-performing model was selected based on ROC-AUC score.
## MLflow Features Used:
- Experiment Tracking
- Parameter Logging
- Metric Logging
- Model Logging
- Run Comparison
## Conclusion:
MLflow successfully tracked all machine learning experiments and enabled comparison of multiple predictive maintenance models. XGBoost achieved the best performance and was selected for deployment and lifecycle management in Lab 14.
# Lab 14
## Model Registry and Lifecycle Management using MLflow
## Objective:
The objective of this lab is to manage machine learning models throughout their lifecycle using MLflow Model Registry, including registration, versioning, staging, production deployment, and rollback.
## Tools and Technologies:
- Python
- VS Code
- Jupyter Notebook
- MLflow
- Scikit-Learn
- XGBoost
## Prerequisite:
Lab 13 completed successfully with trained machine learning models and MLflow experiment tracking.
## Methodology:
1. Connected notebook to MLflow Tracking Server.
2. Generated predictive maintenance dataset.
3. Trained and evaluated:
   - Logistic Regression
   - Random Forest
   - XGBoost
4. Compared model performance using ROC-AUC score.
5. Selected the best-performing model.
6. Registered the model in MLflow Model Registry.
7. Created Version 1 of the registered model.
8. Added model description and metadata tags.
9. Promoted the model to Staging.
10. Tested the model using sample equipment data.
11. Promoted the model to Production.
12. Created a production prediction function.
13. Tested multiple maintenance scenarios.
14. Registered a second model version.
15. Performed model version management.
16. Demonstrated rollback to a previous model version.
## MLflow Features Used:
- Model Registry
- Model Versioning
- Staging Environment
- Production Environment
- Model Metadata
- Model Tags
- Model Lifecycle Management
- Rollback Mechanism
## Registered Model:
PredictiveMaintenance
## Lifecycle Stages:
- None
- Staging
- Production
## Version Management:
- Version 1
- Version 2
## Results:
The best predictive maintenance model was successfully registered and managed through the MLflow lifecycle workflow. Multiple versions were maintained and deployment stages were demonstrated.
## Conclusion:
MLflow Model Registry provides an efficient framework for managing machine learning models from experimentation to production deployment. The lab demonstrated model registration, staging, production deployment, version control, and rollback capabilities.

## Author: Aliha Batool
## Artificial intelligence (AI)
## Lab 13 & 14
