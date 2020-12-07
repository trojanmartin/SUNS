import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import sklearn.decomposition as decomposition
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE


def contains(text, toFind):
    if (toFind in text):
        return 1
    else:
        return 0


def get_owner_value(text):

    splitted = text.split("-")
    return (float(splitted[1]) - float(splitted[0]))/2

def get_genre(genre):
  #  if (contains(genre, "Action") == 1):
  #      return 0
    if (contains(genre, "Strategy") == 1):
        return 1
    if (contains(genre, "Casual") == 1):
        return 2
    if (contains(genre, "Indie") == 1):
        return 3
    if (contains(genre, "RPG") == 1):
        return 4
    if (contains(genre, "Adventure") == 1):
        return 5
    if (contains(genre, "Simulation") == 1):
        return 6
    if (contains(genre, "Sexual Content") == 1):
        return 7
    if (contains(genre, "Free to Play") == 1):
        return 8
    if (contains(genre, "Sports") == 1):
        return 9

    return 10

def eda(data):
    fig = px.scatter(data, x="release_date", y="price")
    fig.update_traces(marker=dict(size=3))
#    fig.show()

    fig = px.scatter(data, x="price", y=data["positive_ratings"] - data["negative_ratings"])
    fig.update_traces(marker=dict(size=3))
    fig.show()

    fig = px.scatter(data, x="release_date", y=data["positive_ratings"])
    fig.update_traces(marker=dict(size=3))
#   fig.show()


    data = clearData(data)
    multiplayer = data.sort_values(by=['release_date']).groupby(['release_date']).sum(['multi_player'])
    fig = px.scatter(x=data['release_date'].unique(), y=multiplayer['multi_player'])
    fig.update_traces(marker=dict(size=3))
    fig.show()

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

    data["genres"] = data["genres"].map(lambda x: get_genre(x))

    data["single_player"] = data["categories"].map(lambda x: contains(x, "Single-player"))
    data["multi_player"] = data["categories"].map(lambda x: contains(x, "Multi-player"))

    data["owners"] = data["owners"].map(lambda x: get_owner_value(x))


    return data

def removeStrings(data):
    data = data.drop(columns=['name','appid', 'developer', 'publisher', 'categories', 'release_date', 'platforms', 'english' ])
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

     sns.countplot(x="cluster_id", hue="genres", data=clustered_data)
     plt.show()

     sns.countplot(x="cluster_id", hue="owners", data=clustered_data)
     plt.show()

     sns.boxplot(x=clustered_data["cluster_id"], y=clustered_data["price"])
     plt.show()

     grouped = clustered_data.groupby(["cluster_id"])
     sns.barplot(x=clustered_data["cluster_id"], y=grouped["average_playtime"].mean())
     plt.show()

     sns.boxplot(x=clustered_data["cluster_id"], y=clustered_data["difficult"])
     plt.show()


def prepare_pca(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = decomposition.PCA(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i: names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels

    return df_matrix

def clustering(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    nClusters = 7
    clusters = cluster.MiniBatchKMeans(n_clusters=nClusters, verbose=True).fit(normalized_data)
#    clusters = cluster.DBSCAN(eps=7,min_samples=25).fit(normalized_data)
    data["cluster_id"] = clusters.labels_
#    create_graphs(data)

    toGraph = prepare_pca(2,normalized_data,clusters.labels_)
    plot_2d(toGraph,clusters.labels_)

#    toGraph = prepare_tsne(3,normalized_data,clusters.labels_)
#    plot_3d(toGraph,clusters.labels_)


def prepare_tsne(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = TSNE(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i: names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels

    return df_matrix


def plot_2d(df, name='labels'):
    fig = px.scatter(df, x='x', y='y',
                        color=name, opacity=0.5)

    fig.update_traces(marker=dict(size=3))
    fig.show()

    print("sad")


def plot_3d(df, name='labels'):
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color=name, opacity=0.5)

    fig.update_traces(marker=dict(size=3))
    fig.show()

    print("sad")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df1 = pd.read_csv("data/steam.csv")
    df2 = pd.read_csv("data/steamspy_tag_data.csv")
    data = pd.merge(df1, df2, left_on='appid', right_on='appid', how='left')

    eda(data)

    data = clearData(data)
    data = removeStrings(data)
    clustering(data)
    print("sad")
