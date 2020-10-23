import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
import tensorflow.keras as kr
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def clear_data(df):
    df = df.dropna(how="any", axis=0)
    df = df.drop(columns=['track.id','track.name','track.artist','track.popularity','track.album.id','track.album.name','track.album.release_date','playlist_name','playlist_id', 'playlist_subgenre', 'duration_ms'])
    df['playlist_genre'] = df['playlist_genre'].map({'edm': 0, 'latin': 1, 'pop': 2, 'r&b': 3, 'rap': 4, 'rock': 5})

    return df

def createGraphs(df):
   fig = px.scatter(df, x="height", y="cardio")
    #fig.show()
    #fig = px.scatter(df, x="weight", y="cardio")
    #fig.show()



if __name__ == '__main__':
    test_df = pd.read_csv("data/test.csv")
    train_df = pd.read_csv("data/train.csv")

    train_df = clear_data(train_df)
    test_df = clear_data(test_df)

#   createGraphs(train_df)

    train, validate = np.split(train_df.sample(frac=1), [int(.8 * len(train_df))])

    train_y = pd.get_dummies(train['playlist_genre'])
    train_x = train.drop(columns='playlist_genre')

    val_y = pd.get_dummies(validate['playlist_genre'])
    val_x = validate.drop(columns='playlist_genre')

    test_y = test_df['playlist_genre']
    test_x = test_df.drop(columns='playlist_genre')

    model = kr.Sequential()
    model.add(kr.layers.Dense(100,input_dim=11, activation="sigmoid"))
    model.add(kr.layers.Dense(50, activation="sigmoid"))
    model.add(kr.layers.Dense(6, activation="sigmoid"))

    model.summary()
    optimizer = kr.optimizers.Adam(learning_rate=0.001)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    training = model.fit(train_x, train_y, epochs=1000, validation_data=(val_x, val_y), callbacks=[early_stopping])

    predicted = np.argmax(model.predict(test_x), axis=1)

    cm =  tf.math.confusion_matrix(test_y,predicted)
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

    print("sad")