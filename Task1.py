import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target)


print("Total rows:",X.shape[0])
print("Total Column:",X.shape[1])


X_temp,X_test,y_temp,y_test=train_test_split(
    X,y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
X_train,X_val,y_train,y_val=train_test_split(
    X_temp,y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)


print("\nTraining data:",len(X_train))
print("Validation data:",len(X_val))
print("Testing data:",len(X_test))

depths=[1,3,4,10,None]
train_scores=[]
val_scores=[]

for depth in depths:
    model=DecisionTreeClassifier(max_depth=depth,random_state=42)
    model.fit(X_train,y_train)

    train_score=accuracy_score(y_train,model.predict(X_train))
    val_score=accuracy_score(y_val,model.predict(X_val))

    train_scores.append(train_score)
    val_scores.append(val_score)

    print(f"\nDepth:{depth}")
    print(f"Training  accuracy:{train_score:.2f}")
    print(f"Validation accuracy:{val_score:.2f}")

depth_labels=[]
for d in depths:
    depth_labels.append(str(d))

plt.plot(depth_labels,train_scores,marker="o",label="Training")
plt.plot(depth_labels,val_scores,marker="o",label="Validation") 
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting")
plt.legend()
plt.show()

model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000))
scores=cross_val_score(model,X_train,y_train,cv=5)

print("\nCross Validation scores:",scores.round(2))
print("Mean accuracy:",round(scores.mean(),2))
print("Standard deviation:",round(scores.std(),2))

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train_wrong=X_scaled[X_train.index]
X_test_wrong=X_scaled[X_test.index]

wrong_model=LogisticRegression(max_iter=5000)
wrong_model.fit(X_train_wrong,y_train)

wrong_accuracy=accuracy_score(
    y_test,wrong_model.predict(X_test_wrong)
)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

correct_model=LogisticRegression(max_iter=5000)
correct_model.fit(X_train_scaled,y_train)

correct_accuracy=accuracy_score(
    y_test,correct_model.predict(X_test_scaled)
)
print("\nWrong method accuracy:",round(wrong_accuracy,2))
print("Correct method accuracy:",round(correct_accuracy,2))
