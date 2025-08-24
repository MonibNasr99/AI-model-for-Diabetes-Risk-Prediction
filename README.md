# AI-model-for-Diabetes-Risk-Prediction
### Executive Summary:
We made this project to tackle an issue in the domain of healthcare section which is to help individuals or the Patients of Diabetes to get quick feedback and to self-diagnose themselves and to be aware of their disease severity and to help them make an early decision regarding their illness.  Using a publicly available dataset, we developed a module to predict if you have Diabetes. Our machine learning model was trained on a comprehensive dataset of medical records and has been rigorously validated for accuracy. It uses the XGBoost algorithm, which is known for its high performance in medical prediction tasks. 
### Problem Statement:
Diabetes is one of the most common chronic diseases worldwide, affecting over 400 million people. Leaving it undiagnosed or untreated, it can lead to serious health complications such as kidney failure, blindness, and cardiovascular diseases. Currently, diabetes diagnosis often relies on lab tests and clinical evaluation. However, many patients are not tested early enough, leading to late-stage detection and complications. A predictive system using patient health data could assist healthcare providers in screening patients more efficiently. Our models have been trained to predict diabetes based on patient data. By identifying risk factors and predicting outcomes, with such capabilities our models can support medical professionals in early intervention and personalized treatment. 
This project has been developed as a machine learning model that predicts whether a patient has diabetes based on medical attributes such as glucose level, BMI, blood pressure, and age. The system will serve as a decision-support tool for healthcare providers, improving early detection and patient outcomes.
### Dataset source:
Our Team has carefully chose a dataset From the Website “Kaggle: Your Machine Learning and Data Science Community” that aligns with our objective and final goal. The dataset is: 
“Pima Indians Diabetes Database”. 
The data set is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The datasets consists of several medical predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
The dataset includes 7 variable columns which are: the number of pregnancies the patient has had, their BMI, insulin level, age, Glucose, BloodPressure, SkinThickness. And it has 769 record (row) of data. 
### Methodology:
The dataset that we chose was already clean and there were no missing values, so when developing the model, we made the decision to use the algorithm the XGBoost algorithm, because is known for its high performance in medical prediction tasks. And we tested the module on other algorithms as well. Which were Logistic Regression, Random Forest and SVM. 

#### Model Development:
https://colab.research.google.com/drive/1vFRVU7enYHCP1qnsIXpSZpktLb0qU-e2?usp=sharing <- Google Colab Code.
[MainCode.ipynb](https://github.com/user-attachments/files/21955708/MainCode.ipynb) <- the source code

#### Evaluation: 
	
 <img width="939" height="732" alt="image" src="https://github.com/user-attachments/assets/5f8a63c6-ae20-4483-9259-0b651e5d5949" />
 
	Model	Accuracy	F1 Score	ROC AUC
 
0	Logistic Regression	0.835443	0.745098	0.887518

1	Random Forest	0.797468	0.680000	0.862845

2	XGBoost	0.810127	0.727273	0.825109

3	SVM	0.797468	0.666667	0.849782


#### Deployment:
https://rashid-diabetes-prediction-xgboost.streamlit.app/ <- the Model deployed
[The Model Deployed Code.py](https://github.com/user-attachments/files/21955729/The.Model.Deployed.Code.py) <- The Model Code.

### Result:
thes are two results from the model: 
First Example:
<img width="1900" height="1016" alt="OT2" src="https://github.com/user-attachments/assets/c782838a-36c2-4b05-bb03-eeb29c1e7348" />
Second Example:
<img width="1896" height="1011" alt="OT1" src="https://github.com/user-attachments/assets/8f957055-4879-4922-9bf4-838642f3ad5e" />


### Demonstration:
to access the model and test it and run it yourself, ->  
https://rashid-diabetes-prediction-xgboost.streamlit.app/ <- Use this link to access the model.
[The Model Deployed Code.py](https://github.com/user-attachments/files/21955729/The.Model.Deployed.Code.py) <- download The Model Code.

## Acknowledgement:
Dataset:https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Libraries: Scikit-learn, XGBoost, Pandas, Streamlit,

Contributors: 
Malaysia - City university - CyberJaya campous.

The members of the Team:

1. 	NASR MOHAMED AHMED MONIB	202409010305
2.	AKBAR RIYAD MOHAMED RASID	202409010654
3.	Muntasir Alsadig Hamid Altahir	202409010773
4.	Ahmed khalid abdelrahman ahmednour	202403010035
5.	Mansoorbasha Abdulkarem Sadeq Noman	202409010644

   
LECTURER Name SIR NAZMIRUL IZZAD BIN NASSIR

SUBJECT CODE: BIT4333

SUBJECT NAME: INTRODUCTION TO MACHINE LEARNING

 
 	SUBMISSION DATE:  28th of August 2025 
