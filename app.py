import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf 
from tensorflow.keras.models import load_model

# Hàm đọc hình ảnh
def read_image(file):
    image = Image.open(file)
    image = np.array(image)
    return image

# Hàm phân đoạn hình ảnh
def predict_segmentation(image, model):
    # Resize hình ảnh về kích thước 256x256
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch
    # Dự đoán
    prediction = model.predict(image)

    # Chuyển đổi dự đoán thành hình ảnh nhị phân
    binary_prediction = (prediction > 0.5).astype(int)

    # Rescale giá trị về đoạn [0, 255] và chuyển định dạng sang uint8
    segmented_image = (binary_prediction[0] * 255).astype(np.uint8)

    return segmented_image

# Hàm dice coefficient
@tf.function
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

# Hàm loss function dice coefficient
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Đăng ký hàm dice_coef_loss
@tf.function
@tf.keras.utils.register_keras_serializable()
def registered_dice_coef_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred)

def main():
    st.title("Image Segmentation App")

    # Upload hình ảnh
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Hiển thị hình ảnh đã upload
        image = read_image(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Nút thực hiện phân đoạn
        if st.button('Segmentation'):
            # Tải mô hình
            model = load_model("final_model.keras", custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
            
            # Thực hiện phân đoạn
            segmented_image = predict_segmentation(image, model)
            
            # Hiển thị hình ảnh sau khi phân đoạn
            st.image(segmented_image, caption='Segmented Image', use_column_width=True)

if __name__ == "__main__":
    main()