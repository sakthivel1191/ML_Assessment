import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier

np.random.seed(42)
n=240
df=pd.DataFrame({
    "income":np.random.randint(25000,120000,n),
    "loan_amount":np.random.randint(10000,80000,n),
    "age":np.random.randint(21,65,n),
    "credit_score":np.random.randint(300,851,n),
    "years_employed":np.random.randint(0,31,n)
})
df["default"]=((df["loan_amount"]/df["income"]>0.65)|(df["credit_score"]<580)).astype(int)
print("DataSet")
print(df.head())

X=df.drop("default",axis=1)
y=df["default"]
X_train,X_val,y_train,y_val=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)
log_model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",LogisticRegression(max_iter=1000))

])

log_model.fit(X_train,y_train)
log_prediction=log_model.predict(X_val)
log_accuracy=accuracy_score(y_val,log_prediction)
print("\nLogistic Regression Accuracy:",round(log_accuracy,2))

print("\nLogistic Regression Prediction:")
print(log_model.predict(X_val.head(5)))
print("\nPrediction Probabilities:")
print(np.round(log_model.predict_proba(X_val.head(5)),2))

tree_model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])
tree_model.fit(X_train,y_train)
tree_prediction=tree_model.predict(X_val)
tree_accuracy=accuracy_score(y_val,tree_prediction)
print("\nDecision Tree Accuracy:",round(tree_accuracy,2))

forest_model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            bootstrap=True,
            oob_score=True,
            random_state=42
    ))
])

forest_model.fit(X_train,y_train)
forest_prediction=forest_model.predict(X_val)
forest_accuracy=accuracy_score(y_val,forest_prediction)
print("\nRandom Forest Accuracy:",round(forest_accuracy,2))
print("Random Forest OOB Score:",round(forest_model.named_steps["classifier"].oob_score_,2))

forest_importance=forest_model.named_steps["classifier"].feature_importances_
forest_feature=pd.DataFrame({
    "Feature":X.columns,
    "Importance":forest_importance
})

forest_feature=forest_feature.sort_values(by="Importance",ascending=False)
print("\nTop 5 Random Forest Features:")
print(forest_feature.head(5))

xgb_model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        early_stopping_rounds=10,
        random_state=42
    ))
])

xgb_model.named_steps["scaler"].fit(X_train)
X_val_scaled= xgb_model.named_steps["scaler"].transform(X_val)
xgb_model.fit(
    X_train,
    y_train,
    classifier__eval_set=[(X_val_scaled,y_val)],
    classifier__verbose=False
)

xgb_prediction=xgb_model.predict(X_val)
xgb_accuracy=accuracy_score(y_val,xgb_prediction)

print("\nXGBoost Accuracy:",round(xgb_accuracy,2))

print("Best XGBoost iteration:",xgb_model.named_steps["classifier"].best_iteration)

xgb_importance=xgb_model.named_steps["classifier"].feature_importances_
xgb_feature=pd.DataFrame({
    "Feature":X.columns,
    "Importance":xgb_importance
})

xgb_feature=xgb_feature.sort_values(by="Importance",ascending=False)

print("\nTop 5 XGBoost Feature:")
print(xgb_feature.head(5))

results=pd.DataFrame({
    "Model":[
        "Logistic Regreesion",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "Validation Score:":[
        log_accuracy,
        tree_accuracy,
        forest_accuracy,
        xgb_accuracy
    ]
})
print("\nClassification Results:")
print(results.round(2).to_string(index=False))

X_reg=df[["income","age","credit_score","years_employed"]]
y_reg=df["loan_amount"]

X_reg_train,X_reg_val,y_reg_train,y_reg_val=train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)
linear_model=Pipeline([
    ("scaler",StandardScaler()),
    ("regressor",LinearRegression())

])

ridge_model=Pipeline([
    ("scaler",StandardScaler()),
    ("regressor",Ridge(alpha=1.0))
])
lasso_model=Pipeline([
    ("scaler",StandardScaler()),
    ("regressor",Lasso(alpha=1000))
])

linear_model.fit(X_reg_train,y_reg_train)
ridge_model.fit(X_reg_train,y_reg_train)
lasso_model.fit(X_reg_train,y_reg_train)

linear_prediction=linear_model.predict(X_reg_val)
ridge_prediction=ridge_model.predict(X_reg_val)
lasso_prediction=lasso_model.predict(X_reg_val)
print("\nRegresion Result(R2 Score):")
print("Linear Regression:",round(r2_score(y_reg_val,linear_prediction),2))
print("Ridge:",round(r2_score(y_reg_val,ridge_prediction),2))
print("Lasso:",round(r2_score(y_reg_val,lasso_prediction),2))

coefficient_table=pd.DataFrame({
    "Feature":X_reg.columns,
    "Linear Regression":linear_model.named_steps["regressor"].coef_,
    "Ridge":ridge_model.named_steps["regressor"].coef_,
    "lasso":lasso_model.named_steps["regressor"].coef_
})
print("\nCoefficient Comparison:")
print(coefficient_table.round(2).to_string(index=False))
zero_count=np.sum(lasso_model.named_steps["regressor"].coef_==0)
print("\nLasso Zero Coefficients:",zero_count)
cluster_data=df[["income","loan_amount","credit_score"]]
scaler=StandardScaler()
cluster_scaled=scaler.fit_transform(cluster_data)

inertia_values=[]
silhouette_values=[]

for k in range (2,9):
    kmeans=KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels=kmeans.fit_predict(cluster_scaled)

    inertia_values.append(kmeans.inertia_)

    score=silhouette_score(cluster_scaled,labels)
    silhouette_values.append(score)
    print("K=",k,"|Inertia=",round(kmeans.inertia_,2),"|Silhouette=",round(score,2))

k_values=list(range(2,9))

plt.plot(k_values,inertia_values,marker="o")
plt.xlabel("Number of Clusters(K)")
plt.ylabel("Inertia")
plt.title("K-Mean Elbow Curve")
plt.grid(True)
plt.show()

plt.plot(k_values,silhouette_values,marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette score")
plt.title("K-Means Silhouette Score")
plt.grid(True)
plt.show()

final_model=KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
)
df["customer_segment"]=final_model.fit_predict(cluster_scaled)

print("\nCustomer Segmenta:")
print(df.groupby("customer_segment")[["income","loan_amount","credit_score"]].mean().round(2))
    