import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_PATH = "model/skinsafe_optimized_model.h5"
IMAGE_SIZE = 128

# MUST be first Streamlit command
st.set_page_config(page_title="Skin Cancer Detector", layout="wide")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={'BinaryFocalCrossentropy': tf.keras.losses.BinaryFocalCrossentropy}
        )
        model.compile()
        return model
    except Exception as e:
        st.error(f"""
            ❌ Model loading failed: {str(e)}
            Possible fixes:
            1. Check if '{MODEL_PATH}' exists
            2. Verify TensorFlow version matches training environment
            3. Ensure image size matches model expectations ({IMAGE_SIZE}x{IMAGE_SIZE})
        """)
        return None

model = load_model()

# --- UI ---
st.title("🔬 Skin Cancer Detection")

with st.sidebar:
    st.markdown("""
    **How to use:**
    1. Upload a skin lesion image
    2. Wait for analysis
    3. Review results (not a medical diagnosis)
    """)
    
    # Threshold adjustment
    threshold = st.slider(
        "Detection Sensitivity",
        min_value=0.1,
        max_value=0.9,
        value=0.3,  # Default lower threshold
        step=0.05,
        help="Lower values catch more malignant cases but may increase false positives"
    )
    
    if os.path.exists("model/labels_map.json"):
        with open("model/labels_map.json") as f:
            st.json(json.load(f))
    
    # Display model metrics if available
    if os.path.exists("model/confusion_matrix.png"):
        st.image("model/confusion_matrix.png", caption="Model Performance")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_column_width=True)
            resized_img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            st.image(resized_img, caption=f"Resized to {IMAGE_SIZE}x{IMAGE_SIZE}", width=200)
            
        with col2:
            img = np.array(resized_img) / 255.0
            
            # Test Time Augmentation
            with st.spinner('Analyzing with multiple augmentations...'):
                datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True
                )
                
                preds = []
                for _ in range(5):  # 5 augmentations
                    augmented = datagen.random_transform(img)
                    pred = model.predict(np.expand_dims(augmented, axis=0), verbose=0)[0][0]
                    preds.append(pred)
                
                pred = float(np.mean(preds))
                confidence = pred if pred >= threshold else (1 - pred)
            
            # Enhanced results display
            if pred >= threshold:
                st.error(f"## 🔴 Suspicious Lesion Detected")
                st.metric("Malignant Probability", f"{pred*100:.1f}%")
                
                if pred >= 0.7:
                    st.warning("""
                    **High Risk Indicators:**
                    - Strongly recommend dermatologist consultation
                    - Monitor for changes in size/color
                    """)
                else:
                    st.warning("""
                    **Moderate Concern:**
                    - Suggest professional evaluation
                    - Photograph monthly for changes
                    """)
            else:
                st.success(f"## 🟢 Likely Benign")
                st.metric("Benign Confidence", f"{(1-pred)*100:.1f}%")
                st.info("""
                **Recommendations:**
                - Regular self-examinations
                - Annual skin checks
                - Sun protection advised
                """)
            
            # Confidence visualization
            st.progress(int(confidence*100))
            
            # Detailed prediction breakdown
            with st.expander("Advanced Analysis"):
                st.markdown("### Prediction Details")
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Raw Prediction Score", f"{pred:.4f}")
                    st.metric("Decision Threshold", f"{threshold:.2f}")
                with col4:
                    st.metric("Benign Score", f"{(1-pred):.4f}")
                    st.metric("Malignant Score", f"{pred:.4f}")
                
                # Visualize prediction distribution
                fig, ax = plt.subplots()
                ax.bar(['Benign', 'Malignant'], [1-pred, pred], color=['green', 'red'])
                ax.set_ylim(0, 1)
                ax.set_title("Prediction Distribution")
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        st.info("Please try a different image or check the file format")

elif uploaded_file and not model:
    st.warning("Model unavailable - cannot process image")

