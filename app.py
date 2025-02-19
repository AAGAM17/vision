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
API_KEY1 = os.getenv("API_KEY1")  # Add second API key

if not API_KEY and not API_KEY1:
    st.error("❌ No API keys found! Check your .env file.")
    st.stop()

# Initialize current API key in session state
if 'current_api_key' not in st.session_state:
    st.session_state.current_api_key = API_KEY

def switch_api_key():
    """Switch between available API keys"""
    if st.session_state.current_api_key == API_KEY and API_KEY1:
        st.session_state.current_api_key = API_KEY1
        return True
    elif st.session_state.current_api_key == API_KEY1 and API_KEY:
        st.session_state.current_api_key = API_KEY
        return True
    return False

def handle_api_response(response_json, retry_func=None, *args, **kwargs):
    """Handle API response and switch keys if needed"""
    if 'error' in response_json:
        error = response_json.get('error', {})
        if isinstance(error, dict):
            error_code = error.get('code')
            error_message = error.get('message', '')
            
            # Check for quota error
            if error_code == 429 and 'quota' in error_message.lower():
                if switch_api_key():
                    st.warning("Switching to alternate API key due to quota limit...")
                    if retry_func:
                        return retry_func(*args, **kwargs)
                    return None
                else:
                    st.error("❌ All API keys have reached their quota limits!")
                    return None
            # Handle other specific error codes
            elif error_code == 400:
                st.error("❌ Bad request: Please check the image format")
                return None
            elif error_code == 401:
                st.error("❌ Authentication failed: Please check your API key")
                return None
            elif error_code == 403:
                st.error("❌ Access forbidden: Please check your API permissions")
                return None
            elif error_code == 500:
                st.error("❌ Server error: Please try again later")
                return None
            else:
                st.error(f"❌ API Error: {error_message}")
                return None
    return response_json

def process_api_response(response, retry_func=None, *args, **kwargs):
    """Process API response and handle errors"""
    try:
        response_json = response.json()
        
        # Check if response is successful and contains choices
        if response.status_code == 200 and "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
            
        # Handle API errors
        handled_response = handle_api_response(response_json, retry_func, *args, **kwargs)
        if handled_response and "choices" in handled_response:
            return handled_response["choices"][0]["message"]["content"]
            
        # If we get here, something went wrong
        error_msg = response_json.get('error', {}).get('message', 'Unknown error occurred')
        return f"❌ API Error: {error_msg}"
        
    except Exception as e:
        return f"❌ Processing Error: {str(e)}"

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
                            "Analyze this hydraulic/pneumatic cylinder engineering drawing carefully.\n"
                            "STRICT RULES:\n"
                            "1) Extract ONLY values that are clearly visible. Return empty string if unclear.\n"
                            "2) Convert all measurements to specified units.\n"
                            "3) For CYLINDER ACTION, determine if SINGLE or DOUBLE action based on design.\n"
                            "4) Look for text labels and dimensions in the drawing.\n"
                            "5) Return data in this EXACT format:\n"
                            "CYLINDER ACTION: [SINGLE/DOUBLE]\n"
                            "BORE DIAMETER: [value] MM\n"
                            "OUTSIDE DIAMETER: [value] MM\n"
                            "ROD DIAMETER: [value] MM\n"
                            "STROKE LENGTH: [value] MM\n"
                            "CLOSE LENGTH: [value] MM\n"
                            "OPEN LENGTH: [value] MM\n"
                            "OPERATING PRESSURE: [value] BAR\n"
                            "OPERATING TEMPERATURE: [value] DEG C\n"
                            "MOUNTING: [value]\n"
                            "ROD END: [value]\n"
                            "FLUID: [value]\n"
                            "DRAWING NUMBER: [value]"
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
        "Authorization": f"Bearer {st.session_state.current_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return process_api_response(response, analyze_cylinder_image, image_bytes)
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
                            "Analyze this valve engineering drawing carefully. Focus on extracting these specific details:\n"
                            "STRICT RULES:\n"
                            "1) For Model No: Look for the complete model number including series and all specifications\n"
                            "2) For Size: Find the nominal size/connection size of the valve (usually in mm or inches)\n"
                            "3) For Pressure Rating: Extract the pressure rating in bar units\n"
                            "4) For Make: Identify the manufacturer name\n"
                            "5) Return data in this EXACT format:\n"
                            "MODEL NO: [Complete model number with all specifications]\n"
                            "SIZE OF VALVE: [Size in mm or inches]\n"
                            "PRESSURE RATING: [Value in BAR]\n"
                            "MAKE: [Manufacturer name]\n\n"
                            "Example output:\n"
                            "MODEL NO: SPVF 25 A 2F 1 A12\n"
                            "SIZE OF VALVE: 25\n"
                            "PRESSURE RATING: 4...12 BAR\n"
                            "MAKE: KRACHT"
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
        "Authorization": f"Bearer {st.session_state.current_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return process_api_response(response, analyze_valve_image, image_bytes)
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
        "Authorization": f"Bearer {st.session_state.current_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return process_api_response(response, analyze_gearbox_image, image_bytes)
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
                            "STRICT RULES:\n"
                            "1. ONLY respond with one of these exact words: CYLINDER, VALVE, or GEARBOX\n"
                            "2. Do not repeat the word or add any other text\n"
                            "3. The response should be exactly one word"
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
        "Authorization": f"Bearer {st.session_state.current_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = process_api_response(response, identify_drawing_type, image_bytes)
        
        if "❌" not in result:
            drawing_type = result.strip().upper()
            if "CYLINDER" in drawing_type:
                return "CYLINDER"
            elif "VALVE" in drawing_type:
                return "VALVE"
            elif "GEARBOX" in drawing_type:
                return "GEARBOX"
            else:
                return f"❌ Invalid drawing type: {drawing_type}"
        return result
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
            "MODEL NO",
            "SIZE OF VALVE",
            "PRESSURE RATING",
            "MAKE"
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
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #0066cc;
            color: white;
        }
        .stSpinner>div {
            text-align: center;
            color: #0066cc;
        }
        .error-text {
            color: #ff0000;
            font-weight: bold;
        }
        .success-text {
            color: #00aa00;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

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

    # File uploader and processing section
    if st.session_state.selected_drawing is None:
        uploaded_file = st.file_uploader("Select File", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                if st.button("Process Drawing", key="process_button"):
                    try:
                        # Step 1: Identify drawing type
                        with st.spinner('Identifying drawing type...'):
                            uploaded_file.seek(0)
                            image_bytes = uploaded_file.read()
                            drawing_type = identify_drawing_type(image_bytes)
                            
                            if not drawing_type or "❌" in drawing_type:
                                st.error(drawing_type if drawing_type else "❌ Could not identify drawing type")
                                return
                            
                            # Initialize new drawing entry
                            new_drawing = {
                                'Drawing Type': drawing_type,
                                'Drawing No.': 'Processing..',
                                'Processing Status': 'Processing..',
                                'Extracted Fields Count': '',
                                'Confidence Score': ''
                            }
                            
                            # Add to table
                            st.session_state.drawings_table = pd.concat([
                                st.session_state.drawings_table,
                                pd.DataFrame([new_drawing])
                            ], ignore_index=True)
                            
                            # Show processing status
                            status_placeholder = st.empty()
                            status_placeholder.info(f"✅ Identified as: {drawing_type}")
                            
                            # Step 2: Process drawing
                            with st.spinner(f'Analyzing {drawing_type.lower()} drawing...'):
                                result = None
                                if drawing_type == "CYLINDER":
                                    result = analyze_cylinder_image(image_bytes)
                                elif drawing_type == "VALVE":
                                    result = analyze_valve_image(image_bytes)
                                elif drawing_type == "GEARBOX":
                                    result = analyze_gearbox_image(image_bytes)
                                
                                if not result or "❌" in result:
                                    st.error(result if result else "❌ Analysis failed")
                                    new_drawing.update({
                                        'Processing Status': 'Failed',
                                        'Confidence Score': '0%',
                                        'Extracted Fields Count': '0/0'
                                    })
                                else:
                                    parsed_results = parse_ai_response(result)
                                    drawing_number = parsed_results.get('DRAWING NUMBER', '')
                                    
                                    if not drawing_number:
                                        st.error("❌ Could not extract drawing number")
                                        new_drawing.update({
                                            'Processing Status': 'Failed',
                                            'Drawing No.': 'Unknown'
                                        })
                                    else:
                                        st.session_state.all_results[drawing_number] = parsed_results
                                        parameters = get_parameters_for_type(drawing_type)
                                        non_empty_fields = sum(1 for k in parameters if parsed_results.get(k, '').strip())
                                        total_fields = len(parameters)
                                        
                                        new_drawing.update({
                                            'Drawing No.': drawing_number,
                                            'Processing Status': 'Completed' if non_empty_fields == total_fields else 'Needs Review!',
                                            'Extracted Fields Count': f"{non_empty_fields}/{total_fields}",
                                            'Confidence Score': f"{(non_empty_fields / total_fields * 100):.0f}%"
                                        })
                                        
                                        status_placeholder.success("✅ Drawing processed successfully!")
                            
                            # Update the table
                            st.session_state.drawings_table.iloc[-1] = new_drawing
                            st.experimental_rerun()
                            
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
            
            with col2:
                st.image(uploaded_file, caption="Uploaded Technical Drawing", use_column_width=True)

    # Display the drawings table
    if not st.session_state.drawings_table.empty:
        st.write("### Processed Drawings")
        
        # Add a View column to the dataframe
        df_with_view = st.session_state.drawings_table.copy()
        
        # Display each row with a view button
        for index, row in df_with_view.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 1])
            
            with col1:
                st.write(row['Drawing Type'])
            with col2:
                st.write(row['Drawing No.'])
            with col3:
                status_color = {
                    'Processing..': 'blue',
                    'Completed': 'green',
                    'Needs Review!': 'orange',
                    'Failed': 'red'
                }.get(row['Processing Status'], 'black')
                st.markdown(f"<span style='color: {status_color}'>{row['Processing Status']}</span>", unsafe_allow_html=True)
            with col4:
                st.write(row['Extracted Fields Count'])
            with col5:
                st.write(row['Confidence Score'])
            with col6:
                if st.button('View', key=f'view_{index}'):
                    st.session_state.selected_drawing = row['Drawing No.']
                    st.experimental_rerun()

    # Show detailed view
    if st.session_state.selected_drawing and st.session_state.selected_drawing in st.session_state.all_results:
        st.write(f"### Detailed View: {st.session_state.selected_drawing}")
        
        results = st.session_state.all_results[st.session_state.selected_drawing]
        drawing_type = st.session_state.drawings_table[
            st.session_state.drawings_table['Drawing No.'] == st.session_state.selected_drawing
        ]['Drawing Type'].iloc[0]
        
        # Initialize session state for edited values if not exists
        if 'edited_values' not in st.session_state:
            st.session_state.edited_values = {}
        
        if st.session_state.selected_drawing not in st.session_state.edited_values:
            st.session_state.edited_values[st.session_state.selected_drawing] = {}
        
        # Create detailed parameters table with editable fields
        parameters = get_parameters_for_type(drawing_type)
        st.write("Edit values that were not detected or need correction:")
        
        # Create columns for the table header
        col1, col2, col3, col4 = st.columns([3, 3, 2, 2])
        with col1:
            st.markdown("**Parameter**")
        with col2:
            st.markdown("**Value**")
        with col3:
            st.markdown("**Confidence**")
        with col4:
            st.markdown("**Status**")
        
        # Display each parameter with editable field
        edited_data = []
        for param in parameters:
            col1, col2, col3, col4 = st.columns([3, 3, 2, 2])
            
            original_value = results.get(param, '')
            # Get the edited value from session state if it exists, otherwise use original
            current_value = st.session_state.edited_values[st.session_state.selected_drawing].get(
                param, 
                original_value
            )
            
            with col1:
                st.write(param)
            
            with col2:
                # Make the field editable
                edited_value = st.text_input(
                    f"Edit {param}",
                    value=current_value,
                    key=f"edit_{param}",
                    label_visibility="collapsed"
                )
                
                # Store edited value in session state if changed
                if edited_value != current_value:
                    st.session_state.edited_values[st.session_state.selected_drawing][param] = edited_value
                
                # Update the value for export
                current_value = edited_value
            
            with col3:
                confidence = "100%" if current_value.strip() else "0%"
                if current_value != original_value and current_value.strip():
                    confidence = "100% (Manual)"
                st.write(confidence)
            
            with col4:
                status = "✅ Auto-filled" if original_value.strip() else "🔴 Manual Required"
                if current_value != original_value and current_value.strip():
                    status = "✅ Manually Filled"
                st.write(status)
            
            # Add to export data
            edited_data.append({
                "Parameter": param,
                "Value": current_value,
                "Confidence": confidence,
                "Status": status
            })
        
        # Add save, export and back buttons
        col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
        with col1:
            if st.button("Back to All Drawings"):
                st.session_state.selected_drawing = None
                st.experimental_rerun()
        
        with col2:
            if st.button("Save Changes"):
                # Update the results with edited values
                for param, value in st.session_state.edited_values[st.session_state.selected_drawing].items():
                    if value.strip():  # Only update non-empty values
                        results[param] = value
                st.session_state.all_results[st.session_state.selected_drawing] = results
                
                # Update the main table statistics
                idx = st.session_state.drawings_table[
                    st.session_state.drawings_table['Drawing No.'] == st.session_state.selected_drawing
                ].index[0]
                
                non_empty_fields = sum(1 for k in parameters if results.get(k, '').strip())
                total_fields = len(parameters)
                
                st.session_state.drawings_table.loc[idx, 'Extracted Fields Count'] = f"{non_empty_fields}/{total_fields}"
                st.session_state.drawings_table.loc[idx, 'Confidence Score'] = f"{(non_empty_fields / total_fields * 100):.0f}%"
                st.session_state.drawings_table.loc[idx, 'Processing Status'] = 'Completed' if non_empty_fields == total_fields else 'Needs Review!'
                
                st.success("✅ Changes saved successfully!")
        
        with col3:
            # Create DataFrame for export
            export_df = pd.DataFrame(edited_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Export to CSV",
                data=csv,
                file_name=f"{st.session_state.selected_drawing}_details.csv",
                mime="text/csv"
            )
        
        with col4:
            # Create a clean format of just the values
            values_text = "\n".join([
                f"{row['Value']}"
                for row in edited_data
                if row['Value'] and row['Value'] != "Not detected"
            ])
            
            if st.button("Copy Values"):
                st.code(values_text, language="text")
                st.toast("✅ Values are ready to copy! Click the copy button in the code block.")
                
                # Also provide a download option for the values
                st.download_button(
                    label="Download Values as TXT",
                    data=values_text,
                    file_name=f"{st.session_state.selected_drawing}_values.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
