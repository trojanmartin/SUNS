import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf


def contains(text, toFind):
    if (toFind in text):
        return 1
    else:
        return 0


def get_owner_value(text):

    splitted = text.split("-")
    return (float(splitted[1]) - float(splitted[0]))/2


def clearData(data):
    data = data.dropna(how="any", axis=0)

    # 1 len jedna /-1
    # 2-5 /-2
    # 5 - 10 /-3
    # 10 > /-4
    # Zaradenie vydavatelov do kotegorii podla poctu vydanych hier
    developer = data["developer"].value_counts()[data["developer"]]
    data["developerCategory"] = developer.values
    data["developerCategory"] = data["developerCategory"].map(lambda x: get_category(x))

    data["windows"]  = data["platforms"].map(lambda x: contains(x,"windows"))
    data["mac"] = data["platforms"].map(lambda x: contains(x, "mac"))
    data["linux"] = data["platforms"].map(lambda x: contains(x, "linux"))

    data["single_player"] = data["categories"].map(lambda x: contains(x, "Single-player"))
    data["multi_player"] = data["categories"].map(lambda x: contains(x, "Multi-player"))

    data["owners"] = data["owners"].map(lambda x: get_owner_value(x))

    return data

def removeStrings(data):
    data = data.drop(columns=['name','appid', 'developer', 'publisher', 'categories', 'genres', 'release_date', 'platforms', 'english' ])
    return data

def get_category(x):
    if (x == 1):
        return 1
    elif (x < 5):
        return 2
    elif (x < 10):
        return 3
    else:
        return 4


def create_graphs(clustered_data):
     plt.figure(figsize=(10, 10))
     sns.countplot(x="cluster_id",hue="developerCategory", data= clustered_data)
     plt.show()

     sns.countplot(x="cluster_id", hue="linux", data=clustered_data)
     plt.show()

     sns.countplot(x="cluster_id", hue="mac", data=clustered_data)
     plt.show()

     sns.countplot(x="cluster_id", hue="windows", data=clustered_data)
     plt.show()

     sns.countplot(x="cluster_id", hue="owners", data=clustered_data)
     plt.show()


     sns.boxplot(x=clustered_data["cluster_id"], y=clustered_data["price"])
     plt.show()

     sns.countplot(x=clustered_data["cluster_id"], y=clustered_data["average_playtime"])
     plt.show()


def clustering(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    nClusters = 7
#    clusters = cluster.MiniBatchKMeans(n_clusters=nClusters, verbose=True).fit(normalized_data)
    clusters = cluster.DBSCAN(eps=7,min_samples=25).fit(normalized_data)

    data["cluster_id"] = clusters.labels_
    create_graphs(data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df1 = pd.read_csv("data/steam.csv")
    df2 = pd.read_csv("data/steamspy_tag_data.csv")
    data = pd.merge(df1, df2, left_on='appid', right_on='appid', how='left')

    data = clearData(data)
    data = removeStrings(data)
    clustering(data)
    print("sad")
