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

# Lab 15: API Deployment & Monitoring using FastAPI and MLflow
## Overview
In modern Machine Learning systems, developing an accurate model is only the first step. To make a model useful in real-world applications, it must be deployed so that external systems can interact with it and obtain predictions in real time. Additionally, monitoring mechanisms must be implemented to track the model's performance, reliability, and operational health.
This lab focused on deploying a machine learning model as a production-ready REST API using FastAPI. The deployed API exposes endpoints for prediction, health monitoring, and performance metrics. Monitoring functionality was also added to track request latency, prediction statistics, and system behavior under load.
The predictive maintenance model developed in previous labs was loaded from the MLflow Model Registry and served through FastAPI. The API was then tested and validated using automated scripts and load testing.
## Lab Objectives
The objectives of this lab were:
Deploy a machine learning model using FastAPI
Load a production model from MLflow Model Registry
Build REST API endpoints
Implement request and response validation
Add health monitoring endpoints
Track API performance metrics
Measure prediction latency
Create a monitoring dashboard
Perform functional and load testing
Validate production readiness of the deployed model
## Technologies Used
Technology	Purpose
Python: Core Programming Language
FastAPI: REST API Development
Uvicorn: ASGI Web Server
MLflow: Model Registry and Model Serving
Pandas: Data Processing
Pydantic: Request Validation
Requests: API Testing
Logging: Monitoring and Diagnostics
## Lab Background
Machine Learning models are commonly deployed behind APIs to allow applications, dashboards, IoT devices, and mobile applications to access prediction services.
A deployed API typically provides:
Prediction Services
Allows users to submit input data and receive predictions.
Health Checks
Provides information about system availability and model readiness.
Monitoring Metrics
Tracks operational statistics such as:
Number of requests
Latency
Error rate
Prediction distribution
These capabilities are critical for maintaining reliable ML systems in production environments.
## Implementation Details
## Part 1: Building the FastAPI Service
## Model Loading
The production model was loaded from the MLflow Model Registry using:
MODEL_NAME = "PredictiveMaintenance"
MODEL_STAGE = "Production"
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.pyfunc.load_model(model_uri)
This ensures that the latest production-ready model is served through the API while maintaining version control and easy model updates.
## Request Schema
The API accepts the following equipment sensor features:
Feature	Description
Temperature	Temperature (°C)
Vibration	Vibration Level
Pressure	Pressure (PSI)
RPM	Rotations Per Minute
Age Days	Days Since Last Maintenance
## Response Schema
The API returns:
Failure prediction status
Failure probability score
Maintenance recommendation
Response latency
Prediction timestamp
## Part 2: API Endpoints
GET /
Returns basic API information and documentation links.
GET /health
Checks API availability and verifies that the production model is successfully loaded.
POST /predict
Accepts equipment sensor data and generates failure predictions.
## Workflow:
Receive sensor readings
Validate request data
Convert input to DataFrame
Generate prediction using ML model
Return prediction results and recommendations
GET /metrics
Provides operational monitoring statistics including:
Total requests
Failure predictions
Failure rate
Average latency
Error count
## Part 3: Monitoring Implementation
A monitoring system was integrated into the API to track operational performance.
Metrics Collected
Total API Requests
Prediction History
Response Latency
Failure Prediction Rate
Error Count
Monitoring Dashboard
A lightweight dashboard (monitor_dashboard.py) was created to display real-time metrics.
## Features:
Automatic refresh every 5 seconds
Live monitoring
Operational visibility
## Displayed Metrics:
Total Requests
Failures Predicted
Failure Rate
Average Latency
Error Count
## Part 4: API Testing
The API was validated using automated test cases implemented in test_api.py.
Test Scenarios
Normal Operation
Low-risk equipment conditions
Expected outcome: Continue operation
High-Risk Operation
Elevated sensor values
Expected outcome: Schedule maintenance
Run tests using:
python test_api.py
## Part 5: Load Testing
Load testing was performed using load_test.py to evaluate API performance under multiple requests.
Configuration
NUM_REQUESTS = 100
Random sensor values were generated for each request.
Performance Metrics
Success Rate
Throughput (Requests/Second)
Average Latency
Minimum Latency
Maximum Latency
Expected Targets
Metric	Target
Success Rate	100%
Average Latency	< 100 ms
Throughput	> 10 req/sec
## Running the Application
Install Dependencies
pip install fastapi uvicorn pydantic requests
Start FastAPI Server
uvicorn api.main:app --reload
Access API
Service	URL
### API	http://localhost:8000
### Swagger UI	http://localhost:8000/docs
FastAPI automatically generates interactive API documentation through Swagger UI.
## Results Achieved
FastAPI application deployed successfully
Production model loaded from MLflow Registry
Request and response schemas implemented
Prediction endpoint operational
Health monitoring endpoint operational
Metrics endpoint operational
Logging and monitoring integrated
Monitoring dashboard implemented
Functional testing completed
Load testing completed
Swagger documentation verified
## Learning Outcomes
Through this lab, I gained practical experience in:
FastAPI development
Model serving with MLflow
REST API design
Request validation using Pydantic
API monitoring and logging
Performance and latency tracking
Load testing techniques
Production MLOps deployment practices
## Conclusion
This lab demonstrated how a machine learning model can be deployed as a production-ready REST API using FastAPI and MLflow. Monitoring, health checks, metrics collection, and load testing were implemented to ensure reliability and observability. The completed system enables real-time predictions while providing visibility into model performance and operational health, representing an important stage in the MLOps lifecycle.

# Lab 16: CI/CD & Automated Retraining
## Overview
This lab focused on completing the MLOps lifecycle by implementing Continuous Integration/Continuous Deployment (CI/CD), data drift detection, and automated model retraining. GitHub Actions was used to automate testing and retraining workflows, while MLflow was used for model registration and version management.
## Objectives
Implement CI/CD using GitHub Actions
Create automated testing workflows
Detect data drift in production data
Generate drift reports
Automate model retraining
Register new model versions in MLflow
Build an end-to-end MLOps pipeline
## Technologies Used
Python
GitHub Actions
MLflow
Scikit-Learn
XGBoost
SciPy
Pandas
NumPy
PyTest
## Project Structure
Lab16/
├── .github/workflows/
│   ├── test.yml
│   └── retrain.yml
├── tests/
│   └── test_model.py
├── drift_detector.py
├── test_drift.py
├── retrain_pipeline.py
├── drift_report.json
└── README.md
## Implementation
CI/CD Pipeline
Implemented GitHub Actions workflows to:
Automatically run tests on code changes
Validate project functionality
Support automated retraining workflows
Data Drift Detection
A drift detection system was developed using the Kolmogorov-Smirnov (KS) Test to compare reference and current data distributions.
Monitored Features:
Temperature
Vibration
Pressure
Generated output:
drift_report.json
Automated Retraining
The retraining pipeline performs:
Generate new training data
Preprocess and split data
Train XGBoost model
Evaluate model performance
Log metrics to MLflow
Register new model version
GitHub Actions Workflow
The retraining workflow:
Runs drift detection
Generates drift reports
Retrains the model
Registers updated models
Uploads artifacts automatically
## Running the Project
Run Drift Detection
python test_drift.py
Run Retraining Pipeline
python retrain_pipeline.py
Run Tests
pytest tests/
## Results Achieved
GitHub Actions CI workflow implemented
Automated testing configured
Data drift detection system developed
Drift reports generated successfully
Automated retraining pipeline created
New models registered in MLflow
Weekly retraining workflow configured
End-to-end MLOps automation completed
## Learning Outcomes
Through this lab, I learned:
CI/CD concepts for Machine Learning
GitHub Actions workflow automation
Data drift detection techniques
Automated model retraining
MLflow model version management
Production MLOps best practices
## Conclusion
This lab completed the final stage of the MLOps lifecycle by integrating CI/CD, drift detection, and automated retraining. The resulting system can continuously monitor data, retrain models when needed, and maintain model performance with minimal manual intervention.

## Author
**Aliha Batool**
### Final Project 4
# Predictive Maintenance MLOps Pipeline
An end-to-end Machine Learning Operations (MLOps) system that includes:
- Data Generation & Training
- Experiment Tracking with MLflow
- Model Registry & Version Control
- FastAPI Model Serving
- API Monitoring & Metrics
- Data Drift Detection
- Automated Retraining
- CI/CD using GitHub Actions

Course: Artificial Intelligence (AI)
