import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
import tensorflow.keras as kr
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def clear_data(df):
    df = df.dropna(how="any", axis=0)
    df = df.drop(columns=['track.id','track.name','track.artist','track.popularity','track.album.id','track.album.name', 'track.album.release_date','playlist_name','playlist_id', 'playlist_subgenre'])
    df['playlist_genre'] = df['playlist_genre'].map({'edm': 0, 'latin': 1, 'pop': 2, 'r&b': 3, 'rap': 4, 'rock': 5})

    return df

def removeOutliers(df):
    df.drop(df[df.tempo > 220].index, inplace=True)
    df.drop(df[df.tempo < 50].index, inplace=True)
    df.drop(df[df.key < 0.3].index, inplace=True)
    df.drop(df[df.loudness < - 30].index, inplace=True)
    df.drop(df[df.speechiness > 0.8].index, inplace=True)
    df.drop(df[df.duration_ms > 1000000].index, inplace=True)

    return df


def normalize(df):
    df['key']  = df['key'] / 11
    df['duration_ms'] = (df['duration_ms'] - df['duration_ms'].min()) / (df['duration_ms'].max() - df['duration_ms'].min())
    df['loudness'] = (df['loudness'] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())
    df['tempo'] = (df['tempo'] - df['tempo'].min())/(df['tempo'].max() - df['tempo'].min())
    return df


def svm(train_df, test_df):
    train = train_df
    train_y = train['playlist_genre']
    train_x = train.drop(columns='playlist_genre')
    test_y = test_df['playlist_genre']
    test_x = test_df.drop(columns='playlist_genre')
    # C_range = np.logspace(-2, 10, 6)
    # param_grid = dict(C=C_range)
    # lr = LogisticRegression()
    # print(lr.get_params().keys())
    # grid = GridSearchCV(SVC(verbose=1,), param_grid=param_grid, verbose=1)
    # grid.fit(train_x, train_y.values.ravel())
    # scores = grid.cv_results_['mean_test_score'].reshape(len(C_range))
    # print("Scores: ", scores)
    # print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
    svc = SVC(C=2.5118864315095797)
    svc.fit(train_x, train_y.values.ravel())
    score = svc.score(test_x, test_y)
    print("Scores: ", score)

def createGraphs(df):
    plt.plot(df['tempo'])
    plt.title('tempo')
    plt.show()
    plt.plot(df['danceability'])
    plt.title('danceability')
    plt.show()
    plt.plot(df['energy'])
    plt.title('energy')
    plt.show()
    plt.plot(df['key'])
    plt.title('key')
    plt.show()
    plt.plot(df['loudness'])
    plt.title('loudness')
    plt.show()
    plt.plot(df['speechiness'])
    plt.title('speechiness')
    plt.show()
    plt.plot(df['acousticness'])
    plt.title('acousticness')
    plt.show()
    plt.plot(df['instrumentalness'])
    plt.title('instrumentalness')
    plt.show()
    plt.plot(df['liveness'])
    plt.title('liveness')
    plt.show()
    plt.plot(df['valence'])
    plt.title('valence')
    plt.show()
    plt.plot(df['duration_ms'])
    plt.title('duration_ms')
    plt.show()


def neural_network(train_df, test_df):
    train, validate = np.split(train_df.sample(frac=1), [int(.8 * len(train_df))])

    train_y = pd.get_dummies(train['playlist_genre'])
    train_x = train.drop(columns='playlist_genre')

    val_y = pd.get_dummies(validate['playlist_genre'])
    val_x = validate.drop(columns='playlist_genre')

    test_y = test_df['playlist_genre']
    test_x = test_df.drop(columns='playlist_genre')

    model = kr.Sequential()
    model.add(kr.layers.Dense(100, input_dim=12, activation="sigmoid"))
    model.add(kr.layers.Dense(20, kernel_regularizer=kr.regularizers.L2(0.01), activation="sigmoid"))
#   model.add(kr.layers.Dense(20, activation="sigmoid"))
#   model.add(kr.layers.Dropout(0.4))
#   model.add(kr.layers.Dense(20, activation="sigmoid"))
    model.add(kr.layers.Dense(6, activation="sigmoid"))

    model.summary()
#    optimizer = kr.optimizers.SGD(0.01)
    optimizer = kr.optimizers.Adam(0.01)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    training = model.fit(train_x, train_y, epochs=1000, validation_data=(val_x, val_y), batch_size=20, callbacks=[early_stopping])

    predicted = np.argmax(model.predict(test_x), axis=1)

    cm = tf.math.confusion_matrix(test_y, predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    # summarize history for accuracy
    plt.plot(training.history['accuracy'])
    plt.plot(training.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    test_df = pd.read_csv("data/test.csv")
    train_df = pd.read_csv("data/train.csv")

    train_df = removeOutliers(train_df)

    train_df = clear_data(train_df)
    test_df = clear_data(test_df)

    train_df = normalize(train_df)
    test_df = normalize(test_df)

#    svm(train_df,test_df)
#    createGraphs(train_df)
#    correlation_matrix = train_df.corr()
#    plt.figure(figsize=(10, 10))
#    sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True, cmap='cubehelix')
#    plt.show()
#    svm(train_df,test_df)
    neural_network(train_df, test_df)

