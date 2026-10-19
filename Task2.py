import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

data={
    "income":[30000,40000,50000,60000,80000,35000,
              45000,70000,55000,90000,25000,65000],
    "loan_amount":[10000,20000,25000,30000,15000,40000,
                   20000,35000,28000,45000,12000,32000],
    "age":[22,25,30,35,28,40,32,45,50,23,42,27],
    "employment":["Salaried","Bussiness","Salaried","Bussiness",
                  "Bussiness","Salaried","Bussiness","Salaried",
                  "Salaried","Bussiness","Salaried","Bussiness"],
    "credit_history":["Good","Poor","Excellent","Good",
                      "Fair","Good","Poor","Excellent",
                      "Good","Excellent","Poor","Fair"],
    "application_date":["10-01-2025","15-02-2025","20-03-2025","12-04-2025",
                        "13-05-2025","25-06-2025","30-07-2025","19-08-2025",
                        "01-09-2025","08-10-2025","27-11-2025","11-12-2025"],
    "default":[0,1,0,0,1,0,1,0,0,0,1,0]
}

df=pd.DataFrame(data)
df.loc[2,"income"]=np.nan
df.loc[4,"employment"]=np.nan
df.loc[6,"credit_history"]=np.nan

print(df.head())

X=df.drop("default",axis=1)
y=df["default"]

def create_features(data):
    data=data.copy()

    date=pd.to_datetime(data["application_date"],format="%d-%m-%Y")
    data["year"]=date.dt.year
    data["month"]=date.dt.month
    data["income_ratio"]=(
        data["income"]/data["loan_amount"])
    data["log_loan"]=np.log1p(data["loan_amount"])
    data=data.drop("application_date",axis=1)

    return data

numeric_column=[
    "income","loan_amount","age",
    "year","month","income_ratio","log_loan"]

categorical_columns=["employment"]
ordinal_columns=["credit_history"]

numeric_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
    ])
categorical_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])
ordinal_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OrdinalEncoder(categories=[["Poor","Fair","Good","Excellent"]]))
])

Preprocessor=ColumnTransformer([
    ("numeric",numeric_pipeline,numeric_column),
    ("categorical",categorical_pipeline,categorical_columns),
    ("ordinal",ordinal_pipeline,ordinal_columns)
])

model=Pipeline([
    ("feature",FunctionTransformer(create_features,validate=False)),
    ("preprocessing",Preprocessor),
    ("classifier",LogisticRegression(max_iter=1000))
])

scores=cross_val_score(
    model,X,y,cv=3,scoring="accuracy"
)

print("\nCross validation scores:",scores)
print("pipeline CV accuracy:",round(scores.mean(),2))

knn_pipeline=Pipeline([
    ("imputer",KNNImputer(n_neighbors=3)),
    ("scaler",StandardScaler())
])
print("\nKNNImputer pipeline created successfully")

model.fit(X,y)
joblib.dump(model,"loan_model.pkl")
print("\n Model saved successfully")

loaded_model=joblib.load("loan_model.pkl")
print("Model loaded successfully")

new_customers=pd.DataFrame([
    {
        "income":50000,
        "loan_amount":20000,
        "age":30,
        "employment":"Salaried",
        "credit_history":"Good",
        "application_date":"01-09-2026"
    },
    {
        "income":80000,
        "loan_amount":25000,
        "age":40,
        "employment":"Salaried",
        "credit_history":"Excellent",
        "application_date":"25-06-2026" 
    },
    {
        "income":25000,
        "loan_amount":40000,
        "age":25,
        "employment":"Bussiness",
        "credit_history":"Poor",
        "application_date":"19-08-2026" 
    }
])

predictions=loaded_model.predict(new_customers)

print("\nPrediction for 3 new rows:",predictions)
print("0=No default")
print("1=Default")