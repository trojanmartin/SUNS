import numpy as np
import pandas as pd
import tensorflow.keras as kr
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from os import system
import sklearn.tree as tree
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier


def clear_data(df):
    df = df.dropna(how="any", axis=0)
    df = df.drop(columns=['objid','specobjid','run','rerun', 'mjd','fiberid'])
    df['class'] = df['class'].map({'STAR': 0, 'GALAXY': 1, 'QSO': 2})
    return df


def neural_network_classification(train_df, test_df):
    train, validate = np.split(train_df.sample(frac=1), [int(.8 * len(train_df))])

    train_y = pd.get_dummies(train['class'])
    train_x = train.drop(columns='class')

    val_y = pd.get_dummies(validate['class'])
    val_x = validate.drop(columns='class')

    test_y = test_df['class']
    test_x = test_df.drop(columns='class')

    model = kr.Sequential()
    model.add(kr.layers.Dense(100, input_dim=11, activation="sigmoid"))
    model.add(kr.layers.Dense(20, activation="sigmoid"))
    model.add(kr.layers.Dense(3, activation="sigmoid"))

    model.summary()
    optimizer = kr.optimizers.Adam(0.01)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    training = model.fit(train_x, train_y, epochs=1000, validation_data=(val_x, val_y), callbacks=[early_stopping])

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
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def forest_classifier(train_df, test_df):
    train_y = train_df['class']
    train_x = train_df.drop(columns='class')

    test_y = test_df['class']
    test_x = test_df.drop(columns='class')

    sampler = RandomUnderSampler(random_state=42)
    train_x, train_y = sampler.fit_sample(train_x,train_y)

    forest_classifier = RandomForestClassifier(max_depth=10,random_state=42,n_estimators=10)
    forest_classifier.fit(train_x,train_y)

    one_tree = forest_classifier.estimators_[1]

    filename = "D:/Trojan/Škola/SUNS/assignment3/tree.dot"
    dotfile = open(filename, 'w')
    tree.export_graphviz(one_tree, out_file=dotfile, feature_names=train_x.columns)
    dotfile.close()
    system("C:/ProgramData/chocolatey/bin/dot.exe -Tpng " + filename + " -o D:/Trojan/Škola/SUNS/assignment3/tree.png")

    pred_y = forest_classifier.predict(test_x)

    cm = tf.math.confusion_matrix(test_y, pred_y)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


def neural_network_reggresion(train_df, test_df):
    train, validate = np.split(train_df.sample(frac=1), [int(.8 * len(train_df))])

    train_y = train[['x_coord', 'y_coord', 'z_coord']]
    train_x = train.drop(columns=['x_coord', 'y_coord', 'z_coord'])
    val_y = validate[['x_coord', 'y_coord', 'z_coord']]
    val_x = validate.drop(columns=['x_coord', 'y_coord', 'z_coord'])
    test_y = test_df[['x_coord', 'y_coord', 'z_coord']]
    test_x = test_df.drop(columns=['x_coord', 'y_coord', 'z_coord'])
#    model = KNeighborsRegressor()
#    model.fit(train_x, train_y)
#    print(model.score(test_x, test_y))

if __name__ == '__main__':
    test_df = pd.read_csv("data/test.csv")
    train_df = pd.read_csv("data/train.csv")

    train_df = clear_data(train_df)
    test_df = clear_data(test_df)
#    neural_network_reggresion(train_df,test_df)
#    correlation_matrix = train_df.corr()
#    plt.figure(figsize=(10, 10))
#    sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True, cmap='cubehelix')
#    plt.show()
#    forest_classifier(train_df,test_df)

    neural_network_classification(train_df, test_df)



