#=============== Source Code Giai Thuat KNN===================
import numpy as np
import pandas as pd
data = pd.read_csv("log2.csv", delimiter=',')
from sklearn.model_selection import train_test_split
X = data.drop(data.columns[[4]], axis=1)
Y = data.Action
total_acc = 0
# Voi k=7 chay 10 vong lap for thay doi gia tri random_state
for b in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3.0, random_state=b*100)
    print("Voi random_state", b*100)
    from sklearn.neighbors import KNeighborsClassifier
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=7)
    Mohinh_KNN.fit(X_train, y_train)
    y_pred = Mohinh_KNN.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("Accuracy is: ", accuracy_score(y_test, y_pred)*100)
    mm = np.unique(y_test)
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, y_pred, labels=mm)
    total_acc = total_acc + accuracy_score(y_test, y_pred)

print(len(X_train))
print(len(X_test))
print("Do chinh xac tong the: ", total_acc) # Do chinh xac tong the la 99.31%

# Thu thay doi cac gia tri K khac nhau voi random_state=500 de xem accuracy thay doi nhu the nao
for a in range(5, 19, 2):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3.0, random_state=500)
    print("Voi K = ", a)
    from sklearn.neighbors import KNeighborsClassifier
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=a)
    Mohinh_KNN.fit(X_train, y_train)
    y_pred = Mohinh_KNN.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("Accuracy is: ", accuracy_score(y_test, y_pred)*100)

# ================Source Code Giai Thuat Bayes Tho Ngay====================
import pandas as pd
data = pd.read_csv("log2.csv", delimiter=',')
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
X = data.drop(data.columns[[4]], axis=1)
Y = data.Action
for b in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3.0, random_state=b*100)
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Voi random_state = ", b*100)
    thucte = y_test
    dubao = model.predict(X_test)
    model.predict_proba(X_test)
    print("Accuracy is: ", accuracy_score(thucte, dubao)*100)
    from sklearn.metrics import confusion_matrix
    cnf_matrix_gnb = confusion_matrix(y_test, dubao)
    print(cnf_matrix_gnb)
print(len(X_train))
print(len(X_test))

# =================Source Code Giai Thuat Cay Quyet Dinh ========================
import pandas as pd
import numpy as np
data = pd.read_csv("log2.csv", delimiter=',')

from sklearn.model_selection import train_test_split
X = data.drop(data.columns[[4]], axis=1)
Y = data.Action
for b in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3.0, random_state=b*100)
    from sklearn.tree import DecisionTreeClassifier
    cqd = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=7, min_samples_leaf=9)
    cqd.fit(X_train, y_train)
    y_pred = cqd.predict(X_test)
    print("Voi random_state = ", b * 100)
    from sklearn.metrics import accuracy_score
    print("Accuracy is: ", accuracy_score(y_test, y_pred) * 100)
    from sklearn.metrics import confusion_matrix
    cnf_matrix_gnb = confusion_matrix(y_test, y_pred)
    print(cnf_matrix_gnb)
print(len(X_train))
print(len(X_test))

