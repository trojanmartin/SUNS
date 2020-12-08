import datetime
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
import sklearn.metrics as metrics
import os

def get_gender(value):
    if(value == "Men"):
        return "Men"

    if (value == "Boys"):
        return "Men"

    if (value == "Women"):
        return "Women"

    if (value == "Girls"):
        return "Women"

def create_model(img_height,img_width,num_classes):
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model


def train(model, train_ds,val_ds,epochs,checkpoint_path):
    log_dir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    print(model.summary())
    training = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback, cp_callback])

    # Summarize history for accuracy
    plt.plot(training.history['accuracy'])
    plt.plot(training.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def load_model(path,model):
    model.load_weights(path)
    return model


if __name__ == '__main__':
    df = pd.read_csv('data/styles.csv', error_bad_lines=False)
    ownImages = pd.read_csv('data/own-styles.csv')
#  df.drop(df[df["gender"] == "Unisex"].index, inplace=True)

    ownImages["id"] = ownImages.apply(lambda row: str(row["id"]) + ".jpg", axis=1)
    df["id"] = df.apply(lambda row: str(row["id"]) + ".jpg", axis=1)

#    df["gender"] = df["gender"].map(lambda value: get_gender(value))

    batch_size = 50
    img_height = 32
    img_width = 32
    epochs = 5
    img_dir = "data/images"
    test_img_dir = "data/own-images"

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255.,
        validation_split=0.2
    )

    test_ds = image_generator.flow_from_dataframe(
        dataframe=ownImages,
        directory=test_img_dir,
        x_col="id",
        y_col="gender",
        target_size=(img_height, img_width)
    )

    train_ds = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=img_dir,
        x_col="id",
        y_col="gender",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset="training"
    )

    val_ds = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=img_dir,
        x_col="id",
        y_col="gender",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset="validation"
    )

    num_classes = 5
    checkpoint_path = "saved/cp.ckpt"

    model = create_model(img_height,img_width,num_classes)

    #train(model,train_ds,val_ds,epochs,checkpoint_path)
    model = load_model(checkpoint_path,model)

    predticted = model.predict(test_ds)

    predicted_classes = numpy.argmax(predticted, axis=1)

    true_classes = test_ds.classes
    class_labels = list(test_ds.class_indices.keys())

    cm = tf.math.confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


    # loss, acc = model.evaluate(test_ds, verbose=2)
   # print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
    print("sad")

