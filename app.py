import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
 
H = 256
W = 256

from metrics import dice_loss, dice_coef

model_path = "model.h5"

model = tf.keras.models.load_model(model_path,custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})

st.set_page_config(
    page_title="Brain Tumor Segmentation App",
    page_icon=":brain:",
    layout="wide"
)

custom_style = """
<style>
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    #MainMenu,
    header,
    footer {
        visibility: hidden;
        height: 0%;
    }
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)


# Function to perform inference
def perform_inference(image):
    original_shape = image.shape[:2]
    original_image = image.copy()
    image = cv2.resize(image, (W, H))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    mask = model.predict(image, verbose=0)[0]
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
    mask = mask >= 0.5
    mask = np.expand_dims(mask, axis=-1)
    segmented_image = original_image * mask
    return original_image, mask, segmented_image

# Function to display images using Matplotlib
def show_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    st.pyplot()

# Function to download sample images
def download_sample_images():
    sample_images = ["1.png", "2.png", "3.png"]  

    for image_name in sample_images:
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_name)
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                st.download_button(
                    label=f"Download {image_name}",
                    data=image_bytes,
                    key=f"download_{image_name}",
                    file_name=image_name,
                    mime="image/jpeg",
                )
        else:
            st.warning(f"Sample image {image_name} not found.")

# Streamlit app
def main():
    st.title("Brain Tumor Segmentation App")

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload a brain scan image...", type=["jpg", "png", "jpeg"])

    st.markdown("""
        Example Instructions:
        - Upload a brain scan image.
        - Or, download sample images below and check the predictions.
    """)

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Perform inference on the uploaded image
        original_image, mask, segmented_image = perform_inference(image)

        # Display images side by side
        st.subheader("Results!")
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Display original image
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # Display mask
        axs[1].imshow(mask.squeeze(), cmap='gray')
        axs[1].set_title("Mask")
        axs[1].axis('off')

        # Display segmented image
        axs[2].imshow(segmented_image)
        axs[2].set_title("Segmented Tumor")
        axs[2].axis('off')

        st.pyplot(fig)
        
if __name__ == "__main__":
    main()

