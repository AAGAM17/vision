import streamlit as st
import base64
from PIL import Image
import io
import pandas as pd
import os
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    st.error("❌ API key not found! Check your .env file.")
    st.stop()  # Stops execution if no API key is found.

# OpenRouter API URL for Qwen2.5-VL-72B-Instruct
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def encode_image_to_base64(image_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")

def parse_ai_response(response_text):
    """Parse the AI response into a structured format. If a value is missing, return an empty string."""
    results = {}
    lines = response_text.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()
            results[key] = value if value else ""  # Keep blank if missing
    return results

def analyze_cylinder_image(image_bytes):
    base64_image = encode_image_to_base64(image_bytes)
    
    payload = {
        "model": "qwen/qwen2.5-vl-72b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze the engineering drawing and extract only the values that are clearly visible in the image.\n"
                            "STRICT RULES:\n"
                            "1) If a value is missing or unclear, return an empty string. DO NOT estimate any values.\n"
                            "2) Convert values to the specified units where applicable.\n"
                            "3) Determine whether the cylinder is SINGLE-ACTION or DOUBLE-ACTION and set it under CYLINDER ACTION.\n"
                            "4) Extract and return data in this format:\n"
                            "CYLINDER ACTION: [value]\n"
                            "BORE DIAMETER: [value] MM\n"
                            "OUTSIDE DIAMETER: \n"
                            "ROD DIAMETER: [value] MM\n"
                            "STROKE LENGTH: [value] MM\n"
                            "CLOSE LENGTH: [value] MM\n"
                            "OPEN LENGTH: \n"
                            "OPERATING PRESSURE: [value] BAR\n"
                            "OPERATING TEMPERATURE: [value] DEG C\n"
                            "MOUNTING: \n"
                            "ROD END: \n"
                            "FLUID: [Determine and Extract] \n"
                            "DRAWING NUMBER: [Extract from Image]"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": base64_image
                    }
                ]
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response_json = response.json()
        
        if response.status_code == 200 and "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        else:
            return f"❌ API Error: {response_json}"  # Returns error details

    except Exception as e:
        return f"❌ Processing Error: {str(e)}"

def main():
    # Set page config
    st.set_page_config(
        page_title="JSW Engineering Drawing DataSheet Extractor",
        layout="wide"
    )

    # Title
    st.title("JSW Engineering Drawing DataSheet Extractor")

    # Define expected parameters in the new order
    parameters = [
        "CYLINDER ACTION",
        "BORE DIAMETER",
        "OUTSIDE DIAMETER",  # Moved below BORE DIAMETER
        "ROD DIAMETER",
        "STROKE LENGTH",
        "CLOSE LENGTH",
        "OPEN LENGTH",  # Kept blank by default
        "OPERATING PRESSURE",
        "OPERATING TEMPERATURE",
        "MOUNTING",  # Kept blank by default
        "ROD END",  # Kept blank by default
        "FLUID",
        "DRAWING NUMBER"  # New field added
    ]

    # File uploader and processing section
    uploaded_file = st.file_uploader("Select File", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if 'results_df' not in st.session_state:
                st.session_state.results_df = None

            if st.button("Process Drawing", key="process_button"):
                with st.spinner('Processing drawing...'):
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    result = analyze_cylinder_image(image_bytes)
                    
                    if "❌ API Error" in result or "❌ Processing Error" in result:
                        st.error(result)
                    else:
                        parsed_results = parse_ai_response(result)
                        st.session_state.results_df = pd.DataFrame([
                            {"Parameter": k, "Value": parsed_results.get(k, "")}  # Blank if missing
                            for k in parameters
                        ])
                        st.success("✅ Drawing processed successfully!")

            if st.session_state.results_df is not None:
                st.write("### Extracted Parameters")
                st.table(st.session_state.results_df)
            
                csv = st.session_state.results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="cylinder_parameters.csv",
                    mime="text/csv"
                )

        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Technical Drawing")

if __name__ == "__main__":
    main()
