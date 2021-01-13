import pandas as pd
from operator import itemgetter
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from xgboost import XGBClassifier
from pathlib import Path

missing_values = ["n/a", "na", "--", "NA", "-"]
Path("confusions").mkdir(parents=True, exist_ok=True)


def prepareData():
    first = pd.read_csv('winequality-red.csv', sep=";")
    first["type"] = 1
    second = pd.read_csv('winequality-white.csv', sep=";")
    second["type"] = 0
    merged = first.append(second, sort=False)
    merged.to_csv('winequality-all.csv', index=False, sep=";")


def generateClassifications(data_name, show_plots=False):
    wine_data = pd.read_csv(f"{data_name}.csv", na_values=missing_values, sep=";")
    print(wine_data.head(10))
    print(25 * '-')
    print(wine_data.describe())
    print(25 * '-')
    if show_plots:
        plt.figure(figsize=(18, 10))
        sns.heatmap(wine_data.corr(), annot=True, cmap=plt.cm.plasma)
        plt.show()
        print(25 * '-')
        plt.figure(figsize=(10, 5))
        sns.countplot(data=wine_data, x='quality').set(xlabel='Ocena', ylabel='Ilość')
        print(wine_data['quality'].value_counts())
        plt.show()
        print(25 * '-')
        wine_data.hist(figsize=(15, 15), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.show()

    print(wine_data.info())
    print(25 * '-')
    print(wine_data.isnull().sum())
    print(25 * '-')
    print(f"Suma błędów: {wine_data.isnull().sum().sum()}")
    print(25 * '-')

    print(wine_data.quality.unique())
    print(wine_data.quality.value_counts())
    # if show_plots:
    #     plt.figure(figsize=(7, 5))
    #     sns.countplot(data=wine_data, x='quality', color='#0504aa').set(xlabel='Ocena', ylabel='Ilość')
    #     plt.show()
    print(25 * '-')
    if show_plots:
        plt.figure(figsize=(10, 8))
        sns.barplot(x='quality', y='alcohol', data=wine_data)
        plt.xlabel('Ocena', fontsize=12)
        plt.ylabel('Alkohol [%]', fontsize=12)
        plt.show()
    print(25 * '-')
    wine_data['quality'] = ['Smaczne' if x >= 7 else 'Niesmaczne' for x in wine_data['quality']]
    label_quality = LabelEncoder()
    wine_data['quality_encoded'] = label_quality.fit_transform(wine_data['quality'])
    print(wine_data.head(10))
    print(25 * '-')
    print(wine_data['quality_encoded'].value_counts())
    print(25 * '-')
    print(wine_data[wine_data['quality_encoded'] == 0].describe())
    print(wine_data[wine_data['quality_encoded'] == 1].describe())
    print(25 * '-')
    if show_plots:
        plt.figure(figsize=(7, 5))
        sns.countplot(data=wine_data, x='quality', color='#0504aa').set(xlabel='Ocena', ylabel='Ilość')
        plt.show()

    inputs = wine_data.drop(['quality', 'quality_encoded'], axis='columns')
    inputs = StandardScaler().fit_transform(inputs)
    quality = wine_data['quality_encoded'].values


    train_inputs, test_inputs, train_quality, test_quality = train_test_split(inputs, quality, train_size=0.67,
                                                                              random_state=1)

    def topologyOne():
        model = Sequential()
        model.add(Dense(units=8, input_dim=inputs.shape[1], activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def topologyTwo():
        model = Sequential()
        model.add(Dense(units=32, input_dim=inputs.shape[1], activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def topologyThree():
        model = Sequential()
        model.add(Dense(units=128, input_dim=inputs.shape[1], activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    knn_classifiers = []
    statistics = []
    classifiers = [
        ("k-NN, k=1", KNeighborsClassifier(n_neighbors=1)),
        ("k-NN, k=3", KNeighborsClassifier(n_neighbors=3)),
        ("k-NN, k=5", KNeighborsClassifier(n_neighbors=5)),
        ("k-NN, k=7", KNeighborsClassifier(n_neighbors=7)),
        ("k-NN, k=11", KNeighborsClassifier(n_neighbors=11)),
        ("Drzewa decyzyjne", DecisionTreeClassifier()),
        ("Naiwny bayes", GaussianNB()),
        ("SVC", SVC(gamma=2, C=1)),
        ("XGB", XGBClassifier(use_label_encoder=False)),
        ('RandomForest', RandomForestClassifier(random_state=1)),
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=100, random_state=0)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=0)),
        ("GradientBoosting",
         GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)),
        ("LogisticRegression", LogisticRegression(max_iter=500, n_jobs=-1)),
        ("Keras - 1", KerasClassifier(build_fn=topologyOne, nb_epoch=200, verbose=0)),
        ("Keras - 2", KerasClassifier(build_fn=topologyTwo, nb_epoch=200, verbose=0)),
        ("Keras - 3", KerasClassifier(build_fn=topologyThree, nb_epoch=200, verbose=0)),
        ("MLP",
         MLPClassifier(hidden_layer_sizes=(13, 13, 13), activation='relu', solver='adam', max_iter=500, verbose=0))
    ]

    for k in range(1, 100):
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(train_inputs, train_quality)
        predict_values = classifier.predict(test_inputs)
        accuracy = metrics.accuracy_score(test_quality, predict_values)
        knn_classifiers.append((k, accuracy))

    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.plot([x[0] for x in knn_classifiers], [x[1] for x in knn_classifiers])
        plt.plot([x[0] for x in knn_classifiers], [x[1] for x in knn_classifiers], '.r')
        plt.xlabel('Ilość sąsiadów')
        plt.ylabel('Przewidziane wartości')
        plt.show()

    best_knn = max(knn_classifiers, key=itemgetter(1))
    classifiers.insert(0, (f"k-NN, k={best_knn[0]}", KNeighborsClassifier(n_neighbors=best_knn[0])))

    for (classifier_name, classifier) in classifiers:
        classifier.fit(train_inputs, train_quality)
        predict_values = classifier.predict(test_inputs)
        accuracy = metrics.accuracy_score(test_quality, predict_values)
        confusion = metrics.confusion_matrix(test_quality, predict_values)
        statistics.append((classifier_name, accuracy))
        print(f"Klasyfikator : {classifier_name}")
        print(metrics.classification_report(test_quality, predict_values))
        # fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(test_quality, predict_values)
        # plt.plot([0, 1], [0, 1], '--k')
        # plt.plot(fpr_rf, tpr_rf)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC')
        # plt.show()
        print(f"Poprawne odpowiedzi {accuracy * 100} %")
        print("Macierz błędu: \n", confusion)
        print(25 * '*')
        if show_plots:
            plot = sns.heatmap(confusion, annot=True, cmap="jet", fmt='g')
            plt.xlabel('Prawdziwe wartości')
            plt.ylabel('Przewidziane wartości')
            plt.title(classifier_name)
            plot.get_figure().savefig(f'confusions/{classifier_name}.png')
            plt.show()

    print(statistics)
    print(f"Najlepszy klasyfikator: {max(statistics, key=itemgetter(1))}")
    print(25 * '*')

    labels = [s[0] for s in statistics]
    accuracies = [round(s[1] * 100, 2) for s in statistics]
    if show_plots:
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.barh(labels, accuracies, color='#0504aa')
        plt.xlim(0, 100)
        ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
        ax.set_title('Porównanie klasyfikatorów', pad=10)

        for i, (value, name) in enumerate(zip(accuracies, labels)):
            ax.text(value, i, value, ha='left')

        plt.show()


# prepareData()
# generateClassifications("winequality-red")
# generateClassifications("winequality-white")
generateClassifications("winequality-all",True)
