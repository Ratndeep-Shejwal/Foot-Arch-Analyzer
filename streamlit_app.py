import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Foot Arch Analyser",
                   layout="centered")
test_dir = "./test"


st.title("Your Foot Doctor")

st.header("Examples")
st.subheader("Taking pictures like these will help you get the best results!")

with st.container():
    demo_imgs = ('./media/NORMAL1.jpg',
                 './media/LOW1.jpg')

    cols = st.columns(2)
    for col_no in range(len(demo_imgs)):
        cols[col_no % len(cols)].image(demo_imgs[col_no],
                                     use_column_width=True)


st.header("Get Your Foot Arch Analysed!")

if(st.button("Load Data and Train Model")):
    st.info("Loading Data")
    pre_df = get_data()
    pre_df["Images"] = pre_focus_crop(pre_df)
    df = pre_df[["Images", "Target"]]
    get_model()

img_file_upload = st.sidebar.file_uploader(label="Upload an image", type=['PNG','JPG'])

img_file_buffer = st.camera_input(label="Take a picture OR Upload an Image from the sidebar",
                                  help="Allow camera access and take a picture like the examples to help us analyse your foot shape for the best results")

if img_file_upload is not None:
    from PIL import Image
    img = np.asarray(Image.open(img_file_upload))
    cv2.imwrite(test_dir+"/NORMAL/test_img.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    from keras.models import load_model
    from keras.preprocessing.image import ImageDataGenerator
    model = load_model("./trainedClassifier.h5")

    # Define the ImageDataGenerator object for test data preprocessing
    test_data_generator = ImageDataGenerator(rescale=1./255)
    batch_size = 32
    image_size = (256, 256)

    # Generate a data iterator for test data using flow_from_directory method
    test_data_iterator = test_data_generator.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical', # or 'binary' for binary classification
    ) # set shuffle to False to preserve the order of predictions

    predictions = model.predict_generator(test_data_iterator)
    st.write("Predicted class probabilities on test set:", predictions)

elif img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8),
                           cv2.IMREAD_COLOR)

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(cv2_img))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    st.write(cv2_img.shape)

    img = cv2.cvtColor(cv2.resize(cv2_img, (1280, 720)), cv2.COLOR_RGB2BGR)

    st.image(img)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask of the skin color regions
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to remove noise and smooth the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=3)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

    # Find contours in the skin mask
    contours, hierarchy = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)

        # Find the bounding box of the contour
        x, y, w, h = cv2.boundingRect(max_contour)

        # Draw the bounding box on the original image
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the image to fit the bounding box
        crop_img = cv2.resize(img[y:y+h, x:x+w],(256,256))

        np.reshape(crop_img,(256, 256, 3))
        st.write(crop_img.shape)

        # Show the result
        st.image(crop_img)
        cv2.imwrite(test_dir+"/NORMAL/test_img.jpg", crop_img)

        from keras.models import load_model
        from keras.preprocessing.image import ImageDataGenerator
        model = load_model("./trainedClassifier.h5")

        # Define the ImageDataGenerator object for test data preprocessing
        test_data_generator = ImageDataGenerator(rescale=1./255)
        batch_size = 32
        image_size = (256, 256)

        # Generate a data iterator for test data using flow_from_directory method
        test_data_iterator = test_data_generator.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', # or 'binary' for binary classification
        ) # set shuffle to False to preserve the order of predictions

        predictions = model.predict_generator(test_data_iterator)
        st.write("Predicted class probabilities on test set:", predictions)
    else:
        st.warning("Cannot identify feet in the image. Please try again with a different one.")