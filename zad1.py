from sklearn.datasets import load_iris
iris = load_iris()

# splitting into train and test datasets


from sklearn.model_selection import train_test_split
datasets = train_test_split(iris.data, iris.target, test_size=0.2)
train_data, test_data, train_labels, test_labels = datasets

# scaling the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# we fit the train data

scaler.fit(train_data)

# scaling the train data

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
print(train_data[:3])

# ///////////////////////////////////////////////


# Training the Model

from sklearn.neural_network import MLPClassifier

# creating an classifier from the model:

mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=3000)

# let's fit the training data to our model

mlp.fit(train_data, train_labels)
from sklearn.metrics import accuracy_score

# predictions_train = mlp.predict(train_data)

# print(accuracy_score(predictions_train, train_labels))

predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))
from sklearn.metrics import confusion_matrix

# confusion_matrix(predictions_train, train_labels)


confusion_matrix(predictions_test, test_labels)
from sklearn.metrics import classification_report
print(classification_report(predictions_test, test_labels))
from sklearn.neural_network import MLPClassifier

# creating an classifier from the model:

mlp3 = MLPClassifier(hidden_layer_sizes=(3,), max_iter=2500)

# let's fit the training data to our model

mlp3.fit(train_data, train_labels)
from sklearn.metrics import accuracy_score

# predictions_train = mlp.predict(train_data)

# print(accuracy_score(predictions_train, train_labels))

predictions_test3 = mlp3.predict(test_data)
print(accuracy_score(predictions_test3, test_labels))
from sklearn.metrics import confusion_matrix

# confusion_matrix(predictions_train, train_labels)


confusion_matrix(predictions_test3, test_labels)

from sklearn.metrics import classification_report

print(classification_report(predictions_test3, test_labels))

from sklearn.neural_network import MLPClassifier

# creating an classifier from the model:

mlp2 = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=3000)

# let's fit the training data to our model

mlp2.fit(train_data, train_labels)

from sklearn.metrics import accuracy_score

# predictions_train = mlp.predict(train_data)

# print(accuracy_score(predictions_train, train_labels))

predictions_test2 = mlp2.predict(test_data)

print(accuracy_score(predictions_test2, test_labels))

from sklearn.metrics import confusion_matrix

# confusion_matrix(predictions_train, train_labels)


confusion_matrix(predictions_test2, test_labels)

from sklearn.metrics import classification_report

print(classification_report(predictions_test2, test_labels))