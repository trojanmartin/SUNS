import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def clearData(df):
    df = df.drop(columns=['id'])

    df['active'].fillna(0, inplace=True)
    df['ap_lo'].fillna(80, inplace=True)

    df['gender'] = df['gender'].map({'man': 1, 'woman': 0})
    df['cholesterol'] = df['cholesterol'].map({'normal': 0,'above normal': 1, 'well above normal': 2})
    df['glucose'] = df['glucose'].map({'normal': 0, 'above normal': 1, 'well above normal': 2})

    #remove row where age is more than 100years
    df.drop(df[df.age > 36500].index, inplace=True)

    #remove outliyers
    df.drop(df[df.ap_hi > 200].index, inplace=True)
    df.drop(df[df.ap_hi < 50].index, inplace=True)

    df.drop(df[df.ap_lo > 190].index, inplace=True)
    df.drop(df[df.ap_hi < 30].index, inplace=True)

    df.drop(df[df.active > 1].index, inplace=True)
    return df
def createGraphs(df):
    fig = px.scatter(df, x="age", y="cardio")
    #fig.show()
    fig = px.scatter(df, x="height", y="cardio")
    #fig.show()
    #fig = px.scatter(df, x="weight", y="cardio")
    #fig.show()
    #fig = px.scatter(df, x="ap_hi", y="cardio")
    #fig.show()
    #fig = px.density_heatmap(df, x="smoke", y="cardio")
    #fig.show()
    #fig = px.scatter(df, x="ap_lo", y="cardio")
    #fig.show()
    #fig = px.density_heatmap(df, x="alco", y="cardio")
    #fig.show()
    #fig = px.density_heatmap(df, x="active", y="cardio")
    #fig.show()
    #fig = px.density_heatmap(df, x="cholesterol", y="cardio")
    #fig.show()
    #fig = px.density_heatmap(df, x="glucose", y="cardio")
    #fig.show()
    #fig = px.density_heatmap(df, x="gender", y="cardio")
    #fig.show()


def binary_classification(df):
    df = df.drop(["height","smoke","alco","active","gender","glucose"],axis=1)
    train, test, validate = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    print(train.shape, test.shape, validate.shape)

    y_train = train['cardio']
    x_train = train.drop(['cardio'], axis=1)
    y_test = test['cardio']
    x_test = test.drop(['cardio'], axis=1)
    y_validate = validate['cardio']
    x_validate = validate.drop(['cardio'], axis=1)

    classifier = MLPClassifier(solver="adam",hidden_layer_sizes=(20,2), alpha=0.0001, tol=0.00001, random_state=1,verbose=True, max_iter=1000)

    classifier.fit(x_train, y_train)


    fig = px.line(classifier.loss_curve_)
    fig.show()

    predictions = classifier.predict(x_test)



    cm = confusion_matrix(predictions, y_test)
    print(cm)

def main():
    df = pd.read_csv("data/srdcove_choroby.csv")
    df = clearData(df)

    correlation_matrix = df.corr()

    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True, cmap='cubehelix')
    #plt.show()

    binary_classification(df)


if __name__ == "__main__":
    main()


