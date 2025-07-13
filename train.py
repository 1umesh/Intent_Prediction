import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
import joblib


df2=pd.read_csv("/Users/umeshmeena/python_projects/lead_intent/customer_conversion_traing_dataset .csv")
df2.drop(columns=["LeadID","ReferralSource","Location","LeadStatus"],inplace=True)
# Sample: Load your real DataFrame df2 here
X = df2.drop(columns=["Conversion (Target)"])
y = df2["Conversion (Target)"]

categorical_cols = ['Gender','LeadSource', 'DeviceType',  'PaymentHistory']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier(
        class_weight='balanced',
        num_leaves=31,
        max_depth=5,
        learning_rate=0.01,
        n_estimators=200
    ))
])

pipeline.fit(X, y)

# joblib.dump(pipeline, "conversion_prediction_pipeline.pkl")
# user_input = pd.DataFrame([{
#     'Age': 29,
#     'Gender': 'Male',
#     'LeadSource': 'Email',
#     'TimeSpent (minutes)': 15,
#     'PagesViewed': 4,
#     'EmailSent': 1,
#     'DeviceType': 'Mobile',
#     'FormSubmissions': 0,
#     'Downloads': 2,
#     'CTR_ProductPage': 0.05,
#     'ResponseTime (hours)': 3,
#     'FollowUpEmails': 1,
#     'SocialMediaEngagement': 5,
#     'PaymentHistory': 'Good'
# }])

# # Predict conversion (0 or 1)
# prediction = pipeline.predict(user_input)
# prediction_proba = pipeline.predict_proba(user_input)

# print(f'Predicted Conversion: {prediction[0]}')
# print(f'Probability: {prediction_proba[0][1]:.4f} (for class 1)')

joblib.dump(pipeline, "conversion_pipeline.pkl")