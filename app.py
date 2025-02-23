import streamlit as st
import base64
from PIL import Image
import io
import pandas as pd
import os
import requests
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Get all API keys from environment variables
API_KEYS = []
for i in range(6):  # Check for API_KEY through API_KEY5
    key = os.getenv(f"API_KEY{i if i > 0 else ''}")
    if key and key.strip():
        API_KEYS.append(key.strip())

if not API_KEYS:
    st.error("❌ No API keys found! Check your .env file.")
    st.stop()

# Initialize API key management in session state
if 'api_key_index' not in st.session_state:
    st.session_state.api_key_index = 0
if 'api_key_usage' not in st.session_state:
    st.session_state.api_key_usage = {key: 0 for key in API_KEYS}

def get_next_api_key():
    """Get the next available API key in rotation"""
    # Get current key index
    current_index = st.session_state.api_key_index
    
    # Try all keys in rotation
    for _ in range(len(API_KEYS)):
        # Move to next key
        next_index = (current_index + 1) % len(API_KEYS)
        st.session_state.api_key_index = next_index
        
        # Get the key
        api_key = API_KEYS[next_index]
        
        # If key hasn't been used too much, use it
        if st.session_state.api_key_usage[api_key] < 10:  # Adjust threshold as needed
            return api_key
            
        # Reset usage count if all keys are heavily used
        if _ == len(API_KEYS) - 1:
            st.session_state.api_key_usage = {key: 0 for key in API_KEYS}
            return API_KEYS[0]
    
    return API_KEYS[0]

def handle_api_response(response_json, retry_func=None, *args, **kwargs):
    """Handle API response and rotate keys if needed"""
    if 'error' in response_json:
        error = response_json.get('error', {})
        if isinstance(error, dict):
            error_code = error.get('code')
            error_message = error.get('message', '').lower()
            current_key = API_KEYS[st.session_state.api_key_index]
            
            # Check for quota/rate limit errors
            if error_code == 429 or 'quota' in error_message or 'rate limit' in error_message:
                # Increment usage count for current key
                st.session_state.api_key_usage[current_key] += 1
                
                # Get next key
                next_key = get_next_api_key()
                if next_key != current_key:
                    st.warning(f"Switching to next API key... (Current key: ...{current_key[-6:]})")
                    if retry_func:
                        return retry_func(*args, **kwargs)
                else:
                    st.error("❌ All API keys have reached their quota limits!")
                    return None
                    
            # Handle other error codes
            elif error_code == 400:
                st.error("❌ Bad request: Please check the image format")
            elif error_code == 401:
                st.error(f"❌ Authentication failed for key ending in ...{current_key[-6:]}")
                # Try next key on auth failure
                next_key = get_next_api_key()
                if retry_func:
                    return retry_func(*args, **kwargs)
            elif error_code == 403:
                st.error("❌ Access forbidden: Please check your API permissions")
            elif error_code == 500:
                st.error("❌ Server error: Please try again later")
            else:
                st.error(f"❌ API Error: {error_message}")
            return None
    return response_json

def process_api_request(func, *args, **kwargs):
    """Process API request with automatic key rotation"""
    # Get current API key
    current_key = API_KEYS[st.session_state.api_key_index]
    
    # Update headers with current key
    headers = {
        "Authorization": f"Bearer {current_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=kwargs.get('payload'))
        response_json = response.json()
        
        # Check if response is successful and contains choices
        if response.status_code == 200 and "choices" in response_json:
            # Increment usage count for successful request
            st.session_state.api_key_usage[current_key] += 1
            return response_json["choices"][0]["message"]["content"]
            
        # Handle API errors
        handled_response = handle_api_response(response_json, func, *args, **kwargs)
        if handled_response and "choices" in handled_response:
            return handled_response["choices"][0]["message"]["content"]
            
        # If we got an error, try with next key
        next_key = get_next_api_key()
        if next_key != current_key:
            return func(*args, **kwargs)
            
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
    """Parse the AI response into a structured format. If a value is missing or contains [value], return an empty string."""
    results = {}
    lines = response_text.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()
            
            # Check if value contains any variation of [value] and set to empty if it does
            if '[value]' in value.lower() or '[values]' in value.lower():
                value = ""
                
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
                            "5) For MOUNTING and ROD END: If value is not clearly visible or is '[value]', return empty string\n"
                            "6) For FLUID: If fluid type is 'HLP', return 'HYD. OIL MINERAL' instead\n"
                            "7) For OPERATING PRESSURE: Look specifically for 'OPERATING PRESSURE' or 'BETRIEBSDRUCK' value.\n"
                            "   DO NOT use nominal pressure or other pressure values. Only use the operating pressure value.\n"
                            "8) For OPERATING TEMPERATURE:\n"
                            "   - If a single value is given (e.g., '60 DEG C'), use that value\n"
                            "   - If a range is given (e.g., '40 TO 50 DEG C' or '-10°C +60°C'), use the maximum value only\n"
                            "   - Always return just the number followed by 'DEG C'\n"
                            "9) Return data in this EXACT format:\n"
                            "CYLINDER ACTION: [SINGLE/DOUBLE]\n"
                            "BORE DIAMETER: [value] MM\n"
                            "ROD DIAMETER: [value] MM\n"
                            "STROKE LENGTH: [value] MM\n"
                            "CLOSE LENGTH: [value] MM\n"
                            "OPERATING PRESSURE: [value from OPERATING PRESSURE/BETRIEBSDRUCK field only] BAR\n"
                            "OPERATING TEMPERATURE: [maximum value if range, single value if no range] DEG C\n"
                            "MOUNTING: [actual mounting type or empty string]\n"
                            "ROD END: [actual rod end type or empty string]\n"
                            "FLUID: [if HLP then 'HYD. OIL MINERAL', else actual fluid type]\n"
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

    return process_api_request(analyze_cylinder_image, image_bytes, payload=payload)

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
                            "Analyze this valve type key diagram carefully. Look at the ordering example and specifications.\n"
                            "STRICT RULES:\n"
                            "1) For Model No: Extract the complete ordering example (e.g. 'SPVF M 25 A 2F 1 A12 ATEX')\n"
                            "2) For Size: Look at the nominal size table in the diagram. Extract BOTH the size number AND unit.\n"
                            "   - Look for values like: DN20, DN25, DN32, etc.\n"
                            "   - Or flow rates like: 90 l/min, 450 l/min, etc.\n"
                            "   - Always include the unit (mm, inches, l/min)\n"
                            "   - Example format: '25 mm' or '90 l/min'\n"
                            "3) For Pressure Rating: Look at the pressure setting range table and include full range in bar\n"
                            "4) For Make: Look at the manufacturer name at top of drawing\n"
                            "5) Return EXACTLY in this format:\n"
                            "MODEL NO: [Full ordering example]\n"
                            "SIZE OF VALVE: [Size with unit (e.g., 25 mm or 90 l/min)]\n"
                            "PRESSURE RATING: [Pressure range] BAR\n"
                            "MAKE: [Manufacturer name]\n\n"
                            "Example outputs:\n"
                            "MODEL NO: SPVF M 25 A 2F 1 A12 ATEX\n"
                            "SIZE OF VALVE: 25 mm (DN25)\n"
                            "PRESSURE RATING: 4...12 BAR\n"
                            "MAKE: KRACHT\n\n"
                            "or\n\n"
                            "MODEL NO: SPVF M 80 A 2F 1 A12\n"
                            "SIZE OF VALVE: 800 l/min\n"
                            "PRESSURE RATING: 10...20 BAR\n"
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

    return process_api_request(analyze_valve_image, image_bytes, payload=payload)

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
                            "2) Extract and return data in this EXACT format:\n"
                            "TYPE: [value]\n"
                            "NUMBER OF TEETH: [value]\n"
                            "MODULE: [value]\n"
                            "MATERIAL: [value]\n"
                            "PRESSURE ANGLE: [value] DEG\n"
                            "FACE WIDTH, LENGTH: [value] MM\n"
                            "HAND: [value]\n"
                            "MOUNTING: [value]\n"
                            "HELIX ANGLE: [value] DEG\n"
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

    return process_api_request(analyze_gearbox_image, image_bytes, payload=payload)

def analyze_nut_image(image_bytes):
    """Analyze nut drawings and extract specific parameters"""
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
                            "Analyze this nut engineering drawing and extract only the values that are clearly visible in the image.\n"
                            "STRICT RULES:\n"
                            "1) If a value is missing or unclear, return an empty string. DO NOT estimate any values.\n"
                            "2) Extract and return data in this EXACT format:\n"
                            "TYPE: [value]\n"
                            "SIZE: [value]\n"
                            "PROPERTY CLASS: [value]\n"
                            "THREAD PITCH: [value]\n"
                            "COATING: [value]\n"
                            "NUT STANDARD: [value]\n"
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

    return process_api_request(analyze_nut_image, image_bytes, payload=payload)

def analyze_lifting_ram_image(image_bytes):
    """Analyze lifting ram drawings and extract specific parameters"""
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
                            "Analyze this lifting ram engineering drawing and extract only the values that are clearly visible in the image.\n"
                            "STRICT RULES:\n"
                            "1) If a value is missing or unclear, return an empty string. DO NOT estimate any values.\n"
                            "2) Extract and return data in this EXACT format:\n"
                            "HEIGHT: [value] MM\n"
                            "TOTAL STROKE: [value] MM\n"
                            "PISTON STROKE: [value] MM\n"
                            "PISTON LIFTING FORCE: [value] KN\n"
                            "WEIGHT: [value] KG\n"
                            "OIL VOLUME: [value] L\n"
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

    return process_api_request(analyze_lifting_ram_image, image_bytes, payload=payload)

def identify_drawing_type(image_bytes):
    """Identify if the drawing is a cylinder, valve, gearbox, nut, or lifting ram"""
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
                            "3. Gearbox\n"
                            "4. Nut\n"
                            "5. Lifting Ram\n\n"
                            "STRICT RULES:\n"
                            "1. ONLY respond with one of these exact words: CYLINDER, VALVE, GEARBOX, NUT, or LIFTING_RAM\n"
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

    return process_api_request(identify_drawing_type, image_bytes, payload=payload)

def get_parameters_for_type(drawing_type):
    """Return the expected parameters for each drawing type"""
    if drawing_type == "CYLINDER":
        return [
            "CYLINDER ACTION",
            "BORE DIAMETER",
            "ROD DIAMETER",
            "STROKE LENGTH",
            "CLOSE LENGTH",
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
            "TYPE",
            "NUMBER OF TEETH",
            "MODULE",
            "MATERIAL",
            "PRESSURE ANGLE",
            "FACE WIDTH, LENGTH",
            "HAND",
            "MOUNTING",
            "HELIX ANGLE",
            "DRAWING NUMBER"
        ]
    elif drawing_type == "NUT":
        return [
            "TYPE",
            "SIZE",
            "PROPERTY CLASS",
            "THREAD PITCH",
            "COATING",
            "NUT STANDARD",
            "DRAWING NUMBER"
        ]
    elif drawing_type == "LIFTING_RAM":
        return [
            "HEIGHT",
            "TOTAL STROKE",
            "PISTON STROKE",
            "PISTON LIFTING FORCE",
            "WEIGHT",
            "OIL VOLUME",
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

    # Title and description - Even more minimal styling
    st.markdown("""
        <div style="text-align: center; padding: 0.25rem 0; border-bottom: 1px solid #ddd; margin-bottom: 1rem;">
            <h1 style="margin: 0; font-size: 1.5rem;">JSW Engineering Drawing DataSheet Extractor</h1>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session states
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
    if 'current_image' not in st.session_state:
        st.session_state.current_image = {}
    if 'edited_values' not in st.session_state:
        st.session_state.edited_values = {}
    if 'processing_queue' not in st.session_state:
        st.session_state.processing_queue = []
    if 'currently_processing' not in st.session_state:
        st.session_state.currently_processing = False

    # File uploader section
    if st.session_state.selected_drawing is None:
        uploaded_files = st.file_uploader("", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

        if uploaded_files:
            for idx, file in enumerate(uploaded_files):
                with st.container():
                    st.markdown("""
                        <style>
                            .drawing-container {
                                border: 1px solid #ddd;
                                border-radius: 4px;
                                padding: 0.5rem;
                                margin-bottom: 0.5rem;
                            }
                            .stButton button {
                                width: 100%;
                                border: 1px solid #ddd !important;
                                border-radius: 4px !important;
                            }
                        </style>
                        <div class="drawing-container">
                    """, unsafe_allow_html=True)
                    
                    cols = st.columns([1, 2, 1])
                    with cols[0]:
                        st.image(file, width=120)  # Even smaller preview
                    
                    with cols[1]:
                        st.markdown(f"<p style='margin: 0; padding: 0.5rem 0;'><strong>{file.name}</strong></p>", unsafe_allow_html=True)
                    
                    with cols[2]:
                        is_processed = False
                        drawing_entry = None
                        for _, row in st.session_state.drawings_table.iterrows():
                            if row['Drawing No.'].endswith(file.name.split('.')[0]):
                                is_processed = True
                                drawing_entry = row
                                break
                        
                        if is_processed:
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("View", key=f"view_{idx}", help="View detailed results"):
                                    st.session_state.selected_drawing = drawing_entry['Drawing No.']
                                    st.experimental_rerun()
                            with col2:
                                if st.button("Copy", key=f"copy_single_{idx}", help="Copy values to clipboard"):
                                    drawing_results = st.session_state.all_results.get(drawing_entry['Drawing No.'], {})
                                    if drawing_results:
                                        values_text = "\t".join([str(v) for v in drawing_results.values() if v and v.strip()])
                                        st.write(f'<textarea id="copy_text_{idx}" style="position: absolute; left: -9999px;">{values_text}</textarea>', unsafe_allow_html=True)
                                        st.markdown(f"""
                                            <script>
                                                var copyText = document.getElementById('copy_text_{idx}');
                                                copyText.select();
                                                document.execCommand('copy');
                                            </script>
                                        """, unsafe_allow_html=True)
                                        st.toast("✅ Copied to clipboard!")
                        else:
                            if st.button("Process", key=f"process_{idx}", help="Process this drawing"):
                                process_drawing(file)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

        # Show processed drawings in a compact table
        if not st.session_state.drawings_table.empty:
            st.markdown("""
                <div style="margin-top: 1rem; border-top: 1px solid #ddd; padding-top: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0;">Processed Drawings</h4>
                </div>
            """, unsafe_allow_html=True)
            
            # Compact table with borders
            st.markdown("""
                <style>
                    .compact-table {
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        padding: 0.5rem;
                    }
                    .compact-row {
                        border-bottom: 1px solid #eee;
                        padding: 0.25rem 0;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            for idx, row in st.session_state.drawings_table.iterrows():
                with st.container():
                    st.markdown('<div class="compact-row">', unsafe_allow_html=True)
                    cols = st.columns([2, 2, 2, 1, 1])
                    with cols[0]:
                        st.write(f"{row['Drawing Type']} - {row['Drawing No.']}")
                    with cols[1]:
                        st.write(row['Processing Status'])
                    with cols[2]:
                        st.write(f"Confidence: {row['Confidence Score']}")
                    with cols[3]:
                        if st.button("View", key=f"view_table_{idx}", help="View detailed results"):
                            st.session_state.selected_drawing = row['Drawing No.']
                            st.experimental_rerun()
                    with cols[4]:
                        if st.button("Copy", key=f"copy_{idx}", help="Copy values to clipboard"):
                            drawing_results = st.session_state.all_results.get(row['Drawing No.'], {})
                            if drawing_results:
                                # Format for Excel (tab-separated)
                                values_text = "\t".join([str(v) for v in drawing_results.values() if v and v.strip()])
                                st.write(f'<textarea id="copy_table_{idx}" style="position: absolute; left: -9999px;">{values_text}</textarea>', unsafe_allow_html=True)
                                st.markdown(f"""
                                    <script>
                                        var copyText = document.getElementById('copy_table_{idx}');
                                        copyText.select();
                                        document.execCommand('copy');
                                    </script>
                                """, unsafe_allow_html=True)
                                st.toast("✅ Copied to clipboard!")
                    st.markdown('</div>', unsafe_allow_html=True)

def process_drawing(file):
    """Process a single drawing file"""
    try:
        file.seek(0)
        image_bytes = file.read()
        
        with st.spinner('Identifying drawing type...'):
            drawing_type = identify_drawing_type(image_bytes)
            if not drawing_type or "❌" in drawing_type:
                st.error(drawing_type if drawing_type else "❌ Could not identify drawing type")
                return

            new_drawing = {
                'Drawing Type': drawing_type,
                'Drawing No.': 'Processing..',
                'Processing Status': 'Processing..',
                'Extracted Fields Count': '',
                'Confidence Score': ''
            }
            
            st.session_state.drawings_table = pd.concat([
                st.session_state.drawings_table,
                pd.DataFrame([new_drawing])
            ], ignore_index=True)
            
            with st.spinner(f'Analyzing {drawing_type.lower()} drawing...'):
                result = analyze_drawing(drawing_type, image_bytes)
                if result and "❌" not in result:
                    update_drawing_results(drawing_type, result, file, new_drawing)
                    st.success(f"✅ Successfully processed {file.name}")
                else:
                    handle_processing_failure(new_drawing)
                    st.error(f"❌ Failed to process {file.name}")
                
                st.session_state.drawings_table.iloc[-1] = new_drawing
                
    except Exception as e:
        st.error(f"❌ Error processing {file.name}: {str(e)}")
    
    st.experimental_rerun()

def analyze_drawing(drawing_type, image_bytes):
    """Analyze drawing based on its type"""
    if drawing_type == "CYLINDER":
        return analyze_cylinder_image(image_bytes)
    elif drawing_type == "VALVE":
        return analyze_valve_image(image_bytes)
    elif drawing_type == "GEARBOX":
        return analyze_gearbox_image(image_bytes)
    elif drawing_type == "NUT":
        return analyze_nut_image(image_bytes)
    elif drawing_type == "LIFTING_RAM":
        return analyze_lifting_ram_image(image_bytes)
    return None

def update_drawing_results(drawing_type, result, file, new_drawing):
    """Update drawing results after successful processing"""
    parsed_results = parse_ai_response(result)
    drawing_number = (parsed_results.get('MODEL NO', '') 
                     if drawing_type == "VALVE" 
                     else parsed_results.get('DRAWING NUMBER', ''))
    
    if not drawing_number or drawing_number == 'Unknown':
        drawing_number = f"{drawing_type}_{len(st.session_state.drawings_table)}"
    
    file.seek(0)
    st.session_state.current_image[drawing_number] = file.read()
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

def handle_processing_failure(new_drawing):
    """Handle drawing processing failure"""
    new_drawing.update({
        'Processing Status': 'Failed',
        'Confidence Score': '0%',
        'Extracted Fields Count': '0/0'
    })

if __name__ == "__main__":
    main()
