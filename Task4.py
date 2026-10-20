import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import(
    train_test_split,GridSearchCV,RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
    accuracy_score,precision_score,recall_score,f1_score,
    confusion_matrix,roc_curve,roc_auc_score,
    mean_absolute_error,mean_squared_error,r2_score,
    silhouette_score
)
from xgboost import XGBClassifier

np.random.seed(42)
n=500

df=pd.DataFrame({
    "income":np.random.randint(25000,120000,n),
    "loan_amount":np.random.randint(10000,80000,n),
    "age":np.random.randint(21,65,n),
    "credit_score":np.random.randint(300,851,n),
    "years_employed":np.random.randint(0,31,n)
})

df["default"]=(
    (df["loan_amount"]/df["income"]>0.8)|
    (df["credit_score"]<520)).astype(int)

print("DataSet")
print(df.head())

print("\nDefault Counts")
print(df["default"].value_counts())

X=df.drop("default",axis=1)
y=df["default"]

X_temp,X_test,y_temp,y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
X_train,X_val,y_train,y_val=train_test_split(
    X_temp,y_temp,test_size=0.25,
    random_state=42,
    stratify=y_temp
)
print("\nTraining rows:",len(X_train))
print("validation rows:",len(X_val))
print("Test rows:",len(X_test))

log_model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",LogisticRegression(max_iter=1000))
])

tree_model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",DecisionTreeClassifier(
        max_depth=5,random_state=42
    ))
])

forest_model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",RandomForestClassifier(
        n_estimators=100,random_state=42
    ))
])

xgb_model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    ))
])

models={
    "Logistic Regression":log_model,
    "Decision Tree":tree_model,
    "Random Forest":forest_model,
    "XGBoost":xgb_model
}

results=[]
fitted_models={}

for name,model in models.items():
    model.fit(X_train,y_train)
    predictions=model.predict(X_val)
    probabilities=model.predict_proba(X_val)[:,1]
    accuracy=accuracy_score(y_val,predictions)
    precision=precision_score(y_val,predictions,zero_division=0)
    recall=recall_score(y_val,predictions,zero_division=0)
    f1=f1_score(y_val,predictions,zero_division=0)
    auc=roc_auc_score(y_val,probabilities)

    results.append({
        "Model":name,
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "ROC-AUC":auc
    })

    fitted_models[name]=model

    print("\nModel:",name)
    print("Accuracy:",round(accuracy,2))
    print("Precision:",round(precision,2))
    print("Recall",round(recall,2))
    print("F1 Score",round(f1,2))
    print("ROC-AUC",round(auc,2))
    print("Confusio Matrix:")
    print(confusion_matrix(y_val,predictions))

results_df=pd.DataFrame(results)
print("\nValidation Results:")
print(results_df.round(2).to_string(index=False))

plt.figure(figsize=(8,6))

for name in ["Logistic Regression","Random Forest","XGBoost"]:
    model=fitted_models[name]
    probabilities=model.predict_proba(X_val)[:,1]

    fpr,tpr,thresholds=roc_curve(y_val,probabilities)
    auc=roc_auc_score(y_val,probabilities)

    plt.plot(fpr,tpr,label=name+"(AUC="+str(round(auc,2))+")")

plt.plot([0,1],[0,1],linestyle="--",label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

best_name=results_df.sort_values("ROC-AUC",ascending=False).iloc[0]["Model"]
best_model=fitted_models[best_name]
best_probabilities=best_model.predict_proba(X_val)[:,1]

threshold_value=np.arange(0.05,0.96,0.01)
chosen_threshold=None
chosen_precision=0
chosen_recall=0

for threshold in threshold_value:
    predictions=(best_probabilities>=threshold).astype(int)
    precision=precision_score(y_val,predictions,zero_division=0)
    recall=recall_score(y_val,predictions,zero_division=0)
    if recall >=0.75  and precision>chosen_precision:
        chosen_threshold=threshold
        chosen_precision=precision
        chosen_recall=recall

print("\nThreshold Selection")
print("Selection model:",best_name)

if chosen_threshold is not None:
    print("Threshold:",round(chosen_threshold,2))
    print("Recall",round(chosen_recall,2))
    print("Precision",round(chosen_precision,2))
else:
    print("No tested threshold reached 75% recall")

rf_pipeline=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",RandomForestClassifier(random_state=42))
])
rf_parameters={
    "classifier__n_estimators":[100,200],
    "classifier__max_depth":[4,8,None],
    "classifier__min_samples_split":[2,5]
}

rf_search=GridSearchCV(
    rf_pipeline,
    rf_parameters,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)
rf_search.fit(X_train,y_train)

print("\nRandom Forest Best Parameters:")
print(rf_search.best_params_)
print("Random Forest Best CV ROC-AUC")
print(round(rf_search.best_score_,2))

xgb_pipeline=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ))
])

xgb_parameters={
    "classifier__n_estimators":[50,100,150,200,300],
    "classifier__max_depth":[2,3,4,5,6],
    "classifier__learning_rate":[0.01,0.05,0.1,0.2]
}

xgb_search= RandomizedSearchCV(
    xgb_pipeline,
    xgb_parameters,
    n_iter=15,
    cv=3,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)
xgb_search.fit(X_train,y_train)
print("\nXGBoost Best Parameter:")
print(xgb_search.best_params_)
print("XGBoost Best CV ROC-AUC:")
print(round(xgb_search.best_score_,2))

tuned_models={
    "Tuned Random Forest":rf_search.best_estimator_,
    "Tuned XGBoost":xgb_search.best_estimator_
}
tuned_result=[]

for name,model in tuned_models.items():
    predictions=model.predict(X_val)
    probabilities=model.predict_proba(X_val)[:,1]

    tuned_result.append({
        "Model":name,
        "Accuracy":accuracy_score(y_val,predictions),
        "Precision":precision_score(y_val,predictions,zero_division=0),
        "Recall":recall_score(y_val,predictions,zero_division=0),
        "F1":f1_score(y_val,predictions,zero_division=0),
        "ROC-AUC":roc_auc_score(y_val,probabilities)
    })
tuned_df=pd.DataFrame(tuned_result)
print("\nTuned Model Result:")
print(tuned_df.round(2).to_string(index=False))

all_results=pd.concat([results_df,tuned_df],ignore_index=True)
all_results=all_results.sort_values("ROC-AUC",ascending=False)
best_name=all_results.iloc[0]["Model"]

if best_name=="Tuned Random Forest":
    final_model=rf_search.best_estimator_
elif best_name=="Tuned XGBoost":
    final_model=xgb_search.best_estimator_
else:
    final_model=fitted_models[best_name]
print("\nBest Model:",best_name)
print("Validation ROC-AUC:",round(all_results.iloc[0]["ROC-AUC"],2))
print("Validation F1:",round(all_results.iloc[0]["F1"],2))

X_reg=df[["income","age","credit_score","years_employed"]]
y_reg=df["loan_amount"]

X_reg_train,X_reg_test,y_reg_train,y_reg_test=train_test_split(
    X_reg,y_reg,
    test_size=0.2,random_state=42
)
regresion_model=Pipeline([
    ("scaler",StandardScaler()),
    ("regressor",LinearRegression())
])
regresion_model.fit(X_reg_train,y_reg_train)
regression_predictions=regresion_model.predict(X_reg_test)

baseline_predictions=np.full(len(y_reg_test),y_reg_train.mean())
model_mae=mean_absolute_error(y_reg_test,regression_predictions)
model_rmse=np.sqrt(mean_squared_error(y_reg_test,regression_predictions))
model_r2=r2_score(y_reg_test,regression_predictions)
baseline_mae=mean_absolute_error(y_reg_test,baseline_predictions)
baseline_rmse=np.sqrt(mean_squared_error(y_reg_test,baseline_predictions))
baseline_r2=r2_score(y_reg_test,baseline_predictions)

print("\nRegression Results")
print("Model MAE:",round(model_mae,2))
print("Model RMSE:",round(model_rmse,2))
print("MOdel R-squared:",round(model_r2,2))

print("\nMean Baseline")
print("Baseline MAE:",round(baseline_mae,2))
print("Baseline RMSE:",round(baseline_rmse,2))
print("Baseline R-squared:",round(baseline_r2,2))

test_predictions=final_model.predict(X_test)
test_probabilities=final_model.predict_proba(X_test)[:,1]

print("\nFINAL TEST RESULTS")
print("Best model:",best_name)
print("Test accuracy:",round(accuracy_score(y_test,test_predictions),2))
print("Test precision:",round(precision_score(y_test,test_predictions,zero_division=0),2))
print("Test recall:",round(recall_score(y_test,test_predictions,zero_division=0),2))
print("Test F1:",round(f1_score(y_test,test_predictions,zero_division=0),2))
print("Test ROC-AUC:",round(roc_auc_score(y_test,test_probabilities),2))

