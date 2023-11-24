import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

X = np.load('D:/Shlok/Work/Smartknower/majorproject/Sign-language-digits-dataset/X.npy')
Y = np.load('D:/Shlok/Work/Smartknower/majorproject/Sign-language-digits-dataset/Y.npy')

def split_dataset(X, Y, test_size=0.25, random_state=1):
    X_conv = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    return train_test_split(X_conv, Y, stratify=Y, test_size=test_size, random_state=random_state)

#define cnn model
def model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

def show_model_history(modelHistory):
    history = pd.DataFrame()
    history["Train Loss"] = modelHistory.history['loss']
    history["Validation Loss"] = modelHistory.history['val_loss']
    history["Train Accuracy"] = modelHistory.history['accuracy']
    history["Validation Accuracy"] = modelHistory.history['val_accuracy']

    fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    axarr[0].set_title("History of Loss in Train and Validation Datasets")
    history[["Train Loss", "Validation Loss"]].plot(ax=axarr[0])
    axarr[1].set_title("History of Accuracy in Train and Validation Datasets")
    history[["Train Accuracy", "Validation Accuracy"]].plot(ax=axarr[1])
    plt.show()

def evaluate_model(X, Y, epochs=200, optimizer=RMSprop(learning_rate=0.0001), callbacks=None):
    X_train, X_test, Y_train, Y_test = split_dataset(X,Y)
    my_model = model()
    my_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    if callbacks is None:
        callbacks = [earlyStopping]
    modelHistory = my_model.fit(X_train, Y_train,validation_data=(X_test, Y_test),callbacks=callbacks,epochs=epochs,verbose=0)

    test_scores = my_model.evaluate(X_test, Y_test, verbose=0)
    train_scores = my_model.evaluate(X_train, Y_train, verbose=0)
    print("[INFO]:Train Accuracy:{:.3f}".format(train_scores[1]))
    print("[INFO]:Validation Accuracy:{:.3f}".format(test_scores[1]))

    show_model_history(modelHistory=modelHistory)
    return model

def predictions():
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
    my_model = model()
    test = np.argmax(Y_test, axis=1)
    pred = np.argmax(my_model.predict(X_test), axis=1)
    return test, pred

# run the test harness for evaluating a model
def run_test_harness():
	# evaluate model
    evaluate_model(X=X, Y=Y)
    # predictions on test data
    test, pred = predictions()
    print('Confusion Matrix')
    print(confusion_matrix(test, pred))
    #classification report
    print('Classification Report')
    print(classification_report(test, pred))

# entry point, run the test harness
run_test_harness()