import pandas as pd
import cv2
import numpy as np
import streamlit as st


def pre_focus_crop(df: pd.DataFrame):
    images = [0]*len(df)
    # Define the lower and upper bounds of the skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    base_dir = "./data/"
    print(df.head())

    for i, imagepath in enumerate(df['Image_Path']):
        try:
            # print(base_dir+imagepath)
            img = cv2.imread(base_dir+imagepath)
        except Exception as e:
            print(e)
        if img is not None:

            # Convert the image to HSV color space
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Create a binary mask of the skin color regions
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # Apply morphological operations to remove noise and smooth the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=3)
            skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

            # Find contours in the skin mask
            contours, hierarchy = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)>0:
            # Find the contour with the largest area
                max_contour = max(contours, key=cv2.contourArea)

                # Find the bounding box of the contour
                x, y, w, h = cv2.boundingRect(max_contour)

                # Draw the bounding box on the original image
                # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Crop the image to fit the bounding box
                crop_img = img[y:y+h, x:x+w]
                images[i] = crop_img
                # df["Images"] = images
            else:
                images[i] = img
                print(i)
    return images

@st.cache_data
def get_data():
    return pd.read_csv("./ImagePath_TargetValue.csv")

@st.cache_resource
def get_model():
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Load the data
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_data = train_datagen.flow_from_directory(
            './data',
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical')

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_data = val_datagen.flow_from_directory(
            './Validation',
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(
            './test',
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical')

    # Define the model
    model = tf.keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', input_shape=(256,256,3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    print(model.summary())

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    print(train_data)
    model.fit(train_data,
            epochs=10,
            validation_data=val_data)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print('Test accuracy:', test_acc)

    # model.save("./trainedClassifier.h5")


# def logReg(df: pd.DataFrame):
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.model_selection import train_test_split

#     X = df['Images']
#     y = df['Target']
#     print(y.shape)
#     print(X[0].shape)
#     X = X.apply(lambda x: cv2.resize(x, (256,256)))
#     X = X.apply(lambda x: x.flatten())
#     print(X[0].shape)
    
#     model = LogisticRegression()
#     X_train, X_test, y_train, y_test = train_test_split(X,y)
#     print(y.shape)
#     print(y_train.shape)
#     print(X_train.shape)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     print(model.score(X_test, y_test))
    
#     from sklearn.metrics import confusion_matrix
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     sns.heatmap(confusion_matrix(y_test,preds), annot=True)
#     plt.show()

if __name__ == '__main__':
    pre_df = get_data()
    pre_df["Images"] = pre_focus_crop(pre_df)
    print(pre_df.head())
    df = pre_df[['Images', 'Target']]
    get_model()
    # logReg(df)

