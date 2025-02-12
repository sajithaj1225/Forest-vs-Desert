import streamlit as st
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from PIL import Image

# Load the model
def load_model():
    model = tf.keras.models.load_model("my_model1.h5")
    return model

model = load_model()

# Set page config
# st.set_page_config(page_title="Background Image", layout="wide")
#
# # Background Image URL (Replace with your own image URL)
# background_image_url = ""  # Replace with your image URL
#
# # Background Image CSS for Full Page and Sidebar
# page_bg_img = f"""
# <style>
#     /* Main App Background */
#     .stApp {{
#         background-image: url("{background_image_url}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }}
#     /* Sidebar Background */
#     [data-testid="stSidebar"] {{
#         background-image: url("{background_image_url}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }}
# </style>
# """
#
# st.markdown(page_bg_img, unsafe_allow_html=True)
#
# import streamlit as st
#
# # Custom CSS to hide the default header
# hide_header_style = """
#     <style>
#         header {visibility: hidden;}
#     </style>
# """
#
# st.markdown(hide_header_style, unsafe_allow_html=True)

# st.title("Streamlit App Without Header")
# st.write("The default Streamlit header is hidden.")

#BACKGROUND COLOR ADDING
import streamlit as st

# Set page config
st.set_page_config(page_title="Different Background Colors", layout="wide")

# CSS for different background colors and removing header
custom_css = """
<style>
    /* Background color for the main app */
    .stApp {
        background-color: #f0f0f0; /* Light Gray */
    }

    /* Background color for the sidebar */
    [data-testid="stSidebar"] {
        background-color: #333333; /* Dark Gray */
    }

    /* Change sidebar text color */
    [data-testid="stSidebar"] * {
        color: white; /* Ensures text remains visible */
    }

    /* Remove header */
    header {
        visibility: hidden;
    }

    /* Remove footer and menu */
    footer, #MainMenu {
        visibility: hidden;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Example content
# st.sidebar.title("Sidebar Menu")
# st.sidebar.write("This sidebar has a dark background.")
#
# st.title("Streamlit with Different Background Colors")
# st.write("The main app has a light background, and the sidebar has a dark background.")


def main():
     # Apply different background colors
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "Details"])
    if page == "Home":
        show_home()
    elif page == "Prediction":
        show_prediction()
    elif page == "Details":
        show_details()



def show_home():
    st.markdown("<h1 style='text-align:center;'>FOREST VS DESSERT</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='font-weight:1000;'> In this project, we developed an AI-driven model designed to classify images as either forests or desserts using a diverse dataset containing hundreds of labeled images from both categories. The model leverages machine learning algorithms to analyze image features such as textures, colors, and patterns to distinguish between natural landscapes and food items. This project demonstrates the power of computer vision in automating image classification tasks, with potential  applications in content organization, recommendation systems, and more..</p>", unsafe_allow_html=True)

    # st.write("""
    #         In this project, we developed an AI-driven model designed to classify images as
    #         either forests or desserts using a diverse dataset containing hundreds of labeled
    #         images from both categories. The model leverages machine learning algorithms to analyze
    #         image features such as textures, colors, and patterns to distinguish between natural landscapes and food items.
    #         This project demonstrates the power of computer vision in automating image classification tasks, with potential
    #         applications in content organization, recommendation systems, and more..
    #         """)


def show_prediction():
    st.markdown("<h2 style='text-align:center;'>Make a Prediction</h2>", unsafe_allow_html=True)
    # Dropdown list options
    options = ['Browse Image','Webcam']

    # Create the dropdown list
    selected_option = st.selectbox('Select an option:', options)

    # Display the selected option
    st.write('You selected:', selected_option)
    if selected_option:
        image = st.file_uploader("Choose an image to predict...", type=["jpg", "jpeg", "png"])



    if image:
        # Read and preprocess the image
        image = imread(image)
        image = resize(image, (150, 150, 1))  # Resize to match model's expected input
        image = image.reshape(1, 150, 150, 1)  # Reshape for the model
        st.image(image,caption="Uploaded Image",width=450)

        # Make prediction
        y_new = model.predict(image)
        ind=y_new.argmax()

        if ind.any() == 0:
            st.write("Prediction: *DESSERT*")
        else:
            st.write("Prediction: *FOREST*")


def show_details():
    st.markdown("<h2 style='text-align:center;'>Model Details</h2>", unsafe_allow_html=True)
    # st.write("""
    # The model is a convolutional neural network (CNN) trained on a dataset of images to classify individuals as either drowsy or natural.
    # It utilizes various layers to extract features and make accurate predictions.
    # - *Input Layer*: Takes images resized to 150x150 pixels.
    # - *Convolutional Layers*: Extract features from images.
    # - *Pooling Layers*: Reduce dimensionality.
    # - *Dense Layers*: Final classification.
    #
    # Ensure to upload clear images for better prediction results.
    # """)

    st.markdown("<h3>References</h3>", unsafe_allow_html=True)
    st.markdown("[Link to google colab](https://colab.research.google.com/drive/1Cl0mW73qqV4MUwskLLECjmzHY1jVjgE3#scrollTo=EoFtiTl0I4x1)",
                unsafe_allow_html=True)
    st.markdown("[Link to Dataset](https://www.kaggle.com/datasets/yasharjebraeily/drowsy-detection-dataset)",
                unsafe_allow_html=True)



main()