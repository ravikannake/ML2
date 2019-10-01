import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

label_encoder = preprocessing.LabelEncoder()
buying = label_encoder.fit_transform(list(data["buying"]))
maint = label_encoder.fit_transform(list(data["maint"]))
door = label_encoder.fit_transform(list(data["door"]))
persons = label_encoder.fit_transform(list(data["persons"]))
lug_boot = label_encoder.fit_transform(list(data["lug_boot"]))
safety = label_encoder.fit_transform(list(data["safety"]))
cls = label_encoder.fit_transform(list(data["cls"]))

X = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,y_train)
acc = knn.score(x_test,y_test)
print(acc)

predicted = knn.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

plot = "buying"
plt.style.use("ggplot")
plt.scatter(data[plot],data["cls"])
plt.xlabel = (plot)
plt.ylabel = ("cls")
plt.show()