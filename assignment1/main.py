import pandas as pd
import plotly.express as px
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np


def clearData(df):
    df = df.drop(columns=['id'])

    df.fillna('active').fillna(0, inplace=True)
    df.fillna('ap_lo').fillna(80, inplace=True)

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

def main():
    df = pd.read_csv("data/srdcove_choroby.csv")
    df = clearData(df)

    correlation_matrix = df.corr()

    sn.heatmap(correlation_matrix,)
    plt.show()


if __name__ == "__main__":
    main()


