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

def analyze_valve_image(image_bytes):
    """Analyze valve drawings and extract specific parameters"""
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
                            "Analyze the valve engineering drawing and extract only the values that are clearly visible in the image.\n"
                            "STRICT RULES:\n"
                            "1) If a value is missing or unclear, return an empty string. DO NOT estimate any values.\n"
                            "2) Extract and return data in this format:\n"
                            "MODEL: [value]\n"
                            "CORRECT MODEL NO: [value]\n"
                            "PRESSURE RATING: [value] BAR\n"
                            "MAKE: [value]\n"
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
            return f"❌ API Error: {response_json}"

    except Exception as e:
        return f"❌ Processing Error: {str(e)}"

def analyze_gearbox_image(image_bytes):
    """Analyze gearbox drawings and extract specific parameters"""
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
                            "Analyze the gearbox engineering drawing and extract only the values that are clearly visible in the image.\n"
                            "STRICT RULES:\n"
                            "1) If a value is missing or unclear, return an empty string. DO NOT estimate any values.\n"
                            "2) Extract and return data in this format:\n"
                            "GEAR RATIO: [value]\n"
                            "SERVICE FACTOR: [value]\n"
                            "INPUT POWER: [value] KW\n"
                            "SHAFT TYPES: [value]\n"
                            "NO OF SHAFT EXTENSIONS: [value]\n"
                            "GEARBOX ORIENTATION: [value]\n"
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
            return f"❌ API Error: {response_json}"

    except Exception as e:
        return f"❌ Processing Error: {str(e)}"

def identify_drawing_type(image_bytes):
    """Identify if the drawing is a cylinder, valve, or gearbox"""
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
                            "Look at this engineering drawing and identify if it is a:\n"
                            "1. Hydraulic/Pneumatic Cylinder\n"
                            "2. Valve\n"
                            "3. Gearbox\n\n"
                            "ONLY respond with one of these exact words: CYLINDER, VALVE, or GEARBOX"
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
            drawing_type = response_json["choices"][0]["message"]["content"].strip().upper()
            return drawing_type
        else:
            return f"❌ API Error: {response_json}"

    except Exception as e:
        return f"❌ Processing Error: {str(e)}"

def get_parameters_for_type(drawing_type):
    """Return the expected parameters for each drawing type"""
    if drawing_type == "CYLINDER":
        return [
            "CYLINDER ACTION",
            "BORE DIAMETER",
            "OUTSIDE DIAMETER",
            "ROD DIAMETER",
            "STROKE LENGTH",
            "CLOSE LENGTH",
            "OPEN LENGTH",
            "OPERATING PRESSURE",
            "OPERATING TEMPERATURE",
            "MOUNTING",
            "ROD END",
            "FLUID",
            "DRAWING NUMBER"
        ]
    elif drawing_type == "VALVE":
        return [
            "MODEL",
            "CORRECT MODEL NO",
            "PRESSURE RATING",
            "MAKE",
            "DRAWING NUMBER"
        ]
    elif drawing_type == "GEARBOX":
        return [
            "GEAR RATIO",
            "SERVICE FACTOR",
            "INPUT POWER",
            "SHAFT TYPES",
            "NO OF SHAFT EXTENSIONS",
            "GEARBOX ORIENTATION",
            "DRAWING NUMBER"
        ]
    return []

def main():
    # Set page config
    st.set_page_config(
        page_title="JSW Engineering Drawing DataSheet Extractor",
        layout="wide"
    )

    # Title
    st.title("JSW Engineering Drawing DataSheet Extractor")

    # Initialize session state
    if 'drawings_table' not in st.session_state:
        st.session_state.drawings_table = pd.DataFrame(columns=[
            'Drawing Type',
            'Drawing No.',
            'Processing Status',
            'Extracted Fields Count',
            'Confidence Score'
        ])
    if 'all_results' not in st.session_state:
        st.session_state.all_results = {}
    if 'selected_drawing' not in st.session_state:
        st.session_state.selected_drawing = None

    # Display the drawings table at the top with view buttons
    if not st.session_state.drawings_table.empty:
        st.write("### Processed Drawings")
        
        # Create a DataFrame copy with a "View" button column
        display_df = st.session_state.drawings_table.copy()
        
        # Display table with view buttons
        for index, row in display_df.iterrows():
            cols = st.columns([2, 2, 2, 2, 2, 1])
            cols[0].write(row['Drawing Type'])
            cols[1].write(row['Drawing No.'])
            cols[2].write(row['Processing Status'])
            cols[3].write(row['Extracted Fields Count'])
            cols[4].write(row['Confidence Score'])
            if cols[5].button('View', key=f'view_{index}'):
                st.session_state.selected_drawing = row['Drawing No.']

    # Show detailed view if a drawing is selected
    if st.session_state.selected_drawing and st.session_state.selected_drawing in st.session_state.all_results:
        st.write(f"### Detailed View: {st.session_state.selected_drawing}")
        
        results = st.session_state.all_results[st.session_state.selected_drawing]
        drawing_type = st.session_state.drawings_table[
            st.session_state.drawings_table['Drawing No.'] == st.session_state.selected_drawing
        ]['Drawing Type'].iloc[0]
        
        # Create detailed parameters table with confidence scores and actions
        detailed_data = []
        parameters = get_parameters_for_type(drawing_type)
        
        for param in parameters:
            value = results.get(param, '')
            confidence = "100%" if value.strip() else "0%"
            if value.strip():
                action = "✅ Auto-filled"
                confidence = "100%"
            elif param in ["CLOSE LENGTH", "OPEN LENGTH"]:
                action = "🔴 Manual Input Required"
                confidence = ""
            else:
                action = "⚠️ Review Required"
                confidence = "95%"
            
            detailed_data.append({
                "Field Name": param,
                "Value": value if value.strip() else "Input box..",
                "Action": action,
                "Confidence Score": confidence
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        
        # Display the detailed table
        st.table(detailed_df)
        
        # Add a button to clear selection
        if st.button("Back to All Drawings"):
            st.session_state.selected_drawing = None
            st.experimental_rerun()

    # File uploader and processing section
    if st.session_state.selected_drawing is None:
        uploaded_file = st.file_uploader("Select File", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                if st.button("Process Drawing", key="process_button"):
                    with st.spinner('Identifying drawing type...'):
                        uploaded_file.seek(0)
                        image_bytes = uploaded_file.read()
                        
                        drawing_type = identify_drawing_type(image_bytes)
                        
                        if "❌" in drawing_type:
                            st.error(drawing_type)
                        else:
                            st.info(f"✅ Identified drawing type: {drawing_type}")
                            
                            new_drawing = {
                                'Drawing Type': drawing_type,
                                'Drawing No.': '',
                                'Processing Status': 'Processing..',
                                'Extracted Fields Count': '',
                                'Confidence Score': ''
                            }
                            
                            with st.spinner(f'Processing {drawing_type.lower()} drawing...'):
                                # Choose the appropriate analysis function based on drawing type
                                if drawing_type == "CYLINDER":
                                    result = analyze_cylinder_image(image_bytes)
                                elif drawing_type == "VALVE":
                                    result = analyze_valve_image(image_bytes)
                                elif drawing_type == "GEARBOX":
                                    result = analyze_gearbox_image(image_bytes)
                                
                                if "❌" in result:
                                    st.error(result)
                                    new_drawing['Processing Status'] = 'Failed'
                                else:
                                    parsed_results = parse_ai_response(result)
                                    drawing_number = parsed_results.get('DRAWING NUMBER', '')
                                    st.session_state.all_results[drawing_number] = parsed_results
                                    
                                    parameters = get_parameters_for_type(drawing_type)
                                    non_empty_fields = sum(1 for k in parameters if parsed_results.get(k, '').strip())
                                    total_fields = len(parameters)
                                    confidence_score = f"{(non_empty_fields / total_fields * 100):.0f}%"
                                    
                                    new_drawing.update({
                                        'Drawing No.': drawing_number,
                                        'Processing Status': 'Completed' if non_empty_fields == total_fields else 'Needs Review!',
                                        'Extracted Fields Count': f"{non_empty_fields}/{total_fields}",
                                        'Confidence Score': confidence_score
                                    })
                                    
                                    st.success("✅ Drawing processed successfully!")
                            
                            # Add new drawing to the table
                            st.session_state.drawings_table = pd.concat([
                                st.session_state.drawings_table,
                                pd.DataFrame([new_drawing])
                            ], ignore_index=True)

            with col2:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Technical Drawing")

if __name__ == "__main__":
    main()
