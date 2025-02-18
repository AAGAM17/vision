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
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None

    # Display the drawings table at the top with view buttons
    if not st.session_state.drawings_table.empty:
        st.write("### Processed Drawings")
        
        # Create the table with alternating row colors and proper styling
        table_html = "<div style='width:100%; background-color:#f0f2f6; padding:10px; border-radius:5px;'>"
        table_html += "<table style='width:100%; border-collapse:collapse;'>"
        table_html += "<tr style='background-color:#e1e4e8;'>"
        table_html += "<th style='padding:10px; text-align:left;'>Drawing Type</th>"
        table_html += "<th style='padding:10px; text-align:left;'>Drawing No.</th>"
        table_html += "<th style='padding:10px; text-align:left;'>Processing Status</th>"
        table_html += "<th style='padding:10px; text-align:left;'>Extracted Fields Count</th>"
        table_html += "<th style='padding:10px; text-align:left;'>Confidence Score</th>"
        table_html += "<th style='padding:10px; text-align:center;'>Action</th>"
        table_html += "</tr>"
        
        for index, row in st.session_state.drawings_table.iterrows():
            bg_color = "#ffffff" if index % 2 == 0 else "#f8f9fa"
            table_html += f"<tr style='background-color:{bg_color};'>"
            table_html += f"<td style='padding:10px;'>{row['Drawing Type']}</td>"
            table_html += f"<td style='padding:10px;'>{row['Drawing No.']}</td>"
            
            # Style the status
            status_color = {
                'Processing..': 'blue',
                'Completed': 'green',
                'Needs Review!': 'orange',
                'Failed': 'red'
            }.get(row['Processing Status'], 'black')
            
            table_html += f"<td style='padding:10px; color:{status_color};'>{row['Processing Status']}</td>"
            table_html += f"<td style='padding:10px;'>{row['Extracted Fields Count']}</td>"
            table_html += f"<td style='padding:10px;'>{row['Confidence Score']}</td>"
            table_html += "<td style='padding:10px; text-align:center;'>"
            
            # Add view button column
            cols = st.columns([5, 1])
            if cols[1].button('View', key=f'view_{index}'):
                st.session_state.selected_drawing = row['Drawing No.']
            
            table_html += "</td></tr>"
        
        table_html += "</table></div>"
        st.markdown(table_html, unsafe_allow_html=True)

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
            
            # Determine action and confidence based on value and parameter type
            if value.strip():
                # Check if the value might be unclear (e.g., very short or unexpected format)
                if len(value) < 2 or (param == 'DRAWING NUMBER' and not any(c.isdigit() for c in value)):
                    action = "⚠️ Review Required"
                    confidence = "95%"
                else:
                    action = "✅ Auto-filled"
                    confidence = "100%"
            else:
                # Specific handling for different parameters
                if param in ["CLOSE LENGTH", "OPEN LENGTH"]:
                    action = "🔴 Manual Input Required"
                    confidence = ""
                    value = "Input box.."
                elif "DIAMETER" in param or "PRESSURE" in param or "TEMPERATURE" in param:
                    action = "🔴 Manual Detection Required (Blur/Unclear)"
                    confidence = "0%"
                    value = "Input box.."
                else:
                    action = "⚠️ Review Required"
                    confidence = "95%"
                    value = "Input box.."
            
            detailed_data.append({
                "Field Name": param,
                "Value": value,
                "Action": action,
                "Confidence Score": confidence
            })
        
        # Create a styled detailed view
        st.markdown("""
            <style>
            .detailed-table th {
                background-color: #e1e4e8;
                padding: 12px;
            }
            .detailed-table td {
                padding: 10px;
            }
            .auto-filled {
                color: green;
            }
            .review-required {
                color: orange;
            }
            .manual-required {
                color: red;
            }
            </style>
        """, unsafe_allow_html=True)
        
        detailed_df = pd.DataFrame(detailed_data)
        st.table(detailed_df.style.apply(lambda x: [
            'background-color: #f8f9fa' if i % 2 == 0 else '' 
            for i in range(len(x))
        ]))
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Back to All Drawings"):
                st.session_state.selected_drawing = None
                st.experimental_rerun()
        
        with col2:
            if st.button("Export to CSV"):
                csv = detailed_df.to_csv(index=False)
                st.download_button(
                    label="Download Detailed Report",
                    data=csv,
                    file_name=f"{st.session_state.selected_drawing}_details.csv",
                    mime="text/csv"
                )

    # File uploader and processing section
    if st.session_state.selected_drawing is None:
        uploaded_file = st.file_uploader("Select File", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                if st.button("Process Drawing", key="process_button"):
                    # Reset processing status
                    st.session_state.processing_status = None
                    
                    with st.spinner('Identifying drawing type...'):
                        uploaded_file.seek(0)
                        image_bytes = uploaded_file.read()
                        
                        # First identify the drawing type
                        drawing_type = identify_drawing_type(image_bytes)
                        
                        if "❌" in drawing_type:
                            st.error(drawing_type)
                        else:
                            # Show initial processing status
                            status_placeholder = st.empty()
                            status_placeholder.info(f"✅ Identified as: {drawing_type}")
                            
                            new_drawing = {
                                'Drawing Type': drawing_type,
                                'Drawing No.': 'Processing..',
                                'Processing Status': 'Processing..',
                                'Extracted Fields Count': '',
                                'Confidence Score': ''
                            }
                            
                            # Add to table immediately to show processing status
                            st.session_state.drawings_table = pd.concat([
                                st.session_state.drawings_table,
                                pd.DataFrame([new_drawing])
                            ], ignore_index=True)
                            
                            with st.spinner(f'Processing {drawing_type.lower()} drawing...'):
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
                                    
                                    # Update the status placeholder
                                    status_placeholder.success("✅ Drawing processed successfully!")
                            
                            # Update the entry in the table
                            st.session_state.drawings_table.iloc[-1] = new_drawing
                            st.experimental_rerun()

            with col2:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Technical Drawing")

if __name__ == "__main__":
    main()
