# ============================================================================================
# FILE: frontend/app.py
# ============================================================================================
import os
import streamlit as st
import requests
from PIL import Image
from dotenv import load_dotenv

# =====================================================================
# CONFIGURATION & SECURITY
# =====================================================================
# Load environment variables from the .env file.
load_dotenv()

# Pull the URL from the environment.
# If it fails to find it, default to the local testing server.
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"
API_URL = os.getenv("API_URL", DEFAULT_API_URL)

# =====================================================================
# UI LAYOUT & BRANDING
# =====================================================================
st.set_page_config(page_title="AI Auto Inspector", layout="wide")

st.markdown("# 🚗 Vehicle Damage Detection System")
st.markdown("### *Automated Deep Learning Inspection Portal*")
st.write("---")

# =====================================================================
# APPLICATION LOGIC
# =====================================================================
uploaded_file = st.file_uploader('Upload a vehicle image for instant structural assessment',
                                 type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    uploaded_file.seek(0)
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Invalid image format. Please upload a valid JPG or PNG.")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Uploaded Asset")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 AI Structural Analysis")

        with st.spinner("Transmitting telemetry to Inference Engine..."):
            try:
                uploaded_file.seek(0)

                # Prepare payload for FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                # Execute network request
                response = requests.post(API_URL, files=files, timeout=45)
                response.raise_for_status()

                # Parse JSON
                result = response.json()

                if "error" in result:
                    st.error(f"Backend processing error: {result['error']}")
                    st.stop()

                prediction = result.get("prediction", "Unknown")

                # Handle confidence score as a float for the progress bar
                raw_confidence = result.get("confidence", 0.0)
                real_confidence = float(raw_confidence) if isinstance(raw_confidence, str) else raw_confidence

                # =====================================================================
                # DYNAMIC RESULTS RENDERING
                # =====================================================================
                if "Normal" in prediction:
                    bg_color = "#d4edda";
                    text_color = "#155724";
                    border_color = "#c3e6cb";
                    status_emoji = "✅ UNDAMAGED"
                elif "Breakage" in prediction:
                    bg_color = "#fff3cd";
                    text_color = "#856404";
                    border_color = "#ffeeba";
                    status_emoji = "⚠️ COMPONENT BROKEN"
                else:
                    bg_color = "#f8d7da";
                    text_color = "#721c24";
                    border_color = "#f5c6cb";
                    status_emoji = "💥 SEVERE CRUSH IMPACT"

                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; color: {text_color}; border: 1px solid {border_color}; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
                        <h2 style="margin: 0; font-size: 24px;">{status_emoji}</h2>
                        <p style="margin: 10px 0 0 0; font-size: 18px; font-weight: bold;">
                            Detected Target State: <span style="text-transform: uppercase;">{prediction}</span>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(f"**Model Confidence Profile: {real_confidence * 100:.1f}%**")
                st.progress(real_confidence)
                st.caption("Live statistical confidence rating generated via Remote FastAPI backend.")

            except requests.exceptions.Timeout:
                st.error(
                    "⏳ **Timeout Error:** The inference engine is currently waking up. Please wait 30 seconds and click upload again.")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"🔌 **Connection Error:** Unable to reach the API at {API_URL}. Please verify the backend is running.")
            except requests.exceptions.HTTPError as http_err:
                st.error(f"🛑 **Server Error:** The API rejected the request. Code: {http_err.response.status_code}")
            except Exception as e:
                st.error(f"⚠️ **System Inference Error:** {e}")