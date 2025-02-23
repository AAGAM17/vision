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
        # Extract payload from kwargs if it exists
        payload = kwargs.get('payload', {})
        
        response = requests.post(API_URL, headers=headers, json=payload)
        response_json = response.json()
        
        # Check if response is successful and contains choices
        if response.status_code == 200 and "choices" in response_json:
            # Increment usage count for successful request
            st.session_state.api_key_usage[current_key] += 1
            return response_json["choices"][0]["message"]["content"]
            
        # Handle API errors
        handled_response = handle_api_response(response_json, func, *args)  # Remove kwargs here
        if handled_response and "choices" in handled_response:
            return handled_response["choices"][0]["message"]["content"]
            
        # If we got an error, try with next key
        next_key = get_next_api_key()
        if next_key != current_key:
            # When retrying with a new key, pass the payload in the same way
            if 'payload' in kwargs:
                return func(*args, payload=kwargs['payload'])
            return func(*args)
            
        # If we get here, something went wrong
        error_msg = response_json.get('error', {}).get('message', 'Unknown error occurred')
        return f"❌ API Error: {error_msg}"
            
    except Exception as e:
        st.error(f"❌ API Error with key ending in ...{current_key[-6:]}")
        # Try with next key on any error
        next_key = get_next_api_key()
        if next_key != current_key:
            # When retrying with a new key, pass the payload in the same way
            if 'payload' in kwargs:
                return func(*args, payload=kwargs['payload'])
            return func(*args)
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

def analyze_cylinder_image(image_bytes, payload=None):
    """Analyze cylinder drawings and extract specific parameters"""
    base64_image = encode_image_to_base64(image_bytes)
    
    if payload is None:
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

def analyze_valve_image(image_bytes, payload=None):
    """Analyze valve drawings and extract specific parameters"""
    base64_image = encode_image_to_base64(image_bytes)
    
    if payload is None:
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

def analyze_gearbox_image(image_bytes, payload=None):
    """Analyze gearbox drawings and extract specific parameters"""
    base64_image = encode_image_to_base64(image_bytes)
    
    if payload is None:
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

def analyze_nut_image(image_bytes, payload=None):
    """Analyze nut drawings and extract specific parameters"""
    base64_image = encode_image_to_base64(image_bytes)
    
    if payload is None:
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

def analyze_lifting_ram_image(image_bytes, payload=None):
    """Analyze lifting ram drawings and extract specific parameters"""
    base64_image = encode_image_to_base64(image_bytes)
    
    if payload is None:
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

def identify_drawing_type(image_bytes, payload=None):
    """Identify if the drawing is a cylinder, valve, gearbox, nut, or lifting ram"""
    base64_image = encode_image_to_base64(image_bytes)
    
    if payload is None:
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
                                "1. ONLY respond with one of these exact words: CYLINDER, VALVE, GEARBOX, NUT, or LIFTING RAM\n"
                                "2. Do not repeat the word or add any other text\n"
                                "3. The response should be exactly one word or two words for 'LIFTING RAM'"
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
    elif drawing_type == "LIFTING RAM":
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

    # Custom CSS for better UI with dark mode support
    st.markdown("""
        <style>
        /* Theme colors - Light Mode */
        [data-theme="light"] {
            --primary-color: #2C3E50;
            --secondary-color: #3498DB;
            --success-color: #27AE60;
            --warning-color: #F39C12;
            --danger-color: #E74C3C;
            --text-color: #2C3E50;
            --text-muted: #95A5A6;
            --bg-light: #F8F9FA;
            --bg-card: #FFFFFF;
            --border-color: #E0E0E0;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Theme colors - Dark Mode */
        [data-theme="dark"] {
            --primary-color: #ECF0F1;
            --secondary-color: #3498DB;
            --success-color: #2ECC71;
            --warning-color: #F1C40F;
            --danger-color: #E74C3C;
            --text-color: #ECF0F1;
            --text-muted: #BDC3C7;
            --bg-light: #2C3E50;
            --bg-card: #34495E;
            --border-color: #4A5568;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        /* Global styles */
        .main {
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            font-family: 'Inter', sans-serif;
            color: var(--text-color);
        }

        /* Card containers */
        .card {
            background: var(--bg-card);
            border-radius: 12px;
            box-shadow: var(--shadow);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            border: 1px solid var(--border-color);
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-color);
        }

        p, span, div {
            color: var(--text-color);
        }

        .text-muted {
            color: var(--text-muted) !important;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, var(--secondary-color), #2980B9);
            color: white !important;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 44px;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
            background: linear-gradient(135deg, #2980B9, #2573a7);
        }

        .stButton>button:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        /* Primary button */
        .stButton.primary>button {
            background: linear-gradient(135deg, #3498DB, #2980B9);
        }

        /* Secondary button */
        .stButton.secondary>button {
            background: linear-gradient(135deg, #95A5A6, #7F8C8D);
        }

        /* Success button */
        .stButton.success>button {
            background: linear-gradient(135deg, #2ECC71, #27AE60);
        }

        /* Warning button */
        .stButton.warning>button {
            background: linear-gradient(135deg, #F1C40F, #F39C12);
        }

        /* Danger button */
        .stButton.danger>button {
            background: linear-gradient(135deg, #E74C3C, #C0392B);
        }

        /* Process button specific styling */
        button[key^="process_"] {
            background: linear-gradient(135deg, #3498DB, #2980B9) !important;
            color: white !important;
            font-weight: 600 !important;
            min-width: 150px;
        }

        /* View button specific styling */
        button[key^="view_"] {
            background: linear-gradient(135deg, #2ECC71, #27AE60) !important;
            color: white !important;
            font-weight: 600 !important;
            min-width: 100px;
        }

        /* Back button specific styling */
        .back-button {
            background: linear-gradient(135deg, #95A5A6, #7F8C8D) !important;
            color: white !important;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        }

        .back-button:hover {
            transform: translateY(-2px);
            background: linear-gradient(135deg, #7F8C8D, #6C7A7A) !important;
        }

        /* Button container styling */
        .button-container {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }

        /* Make sure text in buttons is always white */
        .stButton>button>div {
            color: white !important;
        }

        .stButton>button>div>p {
            color: white !important;
        }

        /* Ensure button text remains visible in both modes */
        [data-theme="light"] .stButton>button,
        [data-theme="dark"] .stButton>button {
            color: white !important;
        }

        [data-theme="light"] .stButton>button>div,
        [data-theme="dark"] .stButton>button>div {
            color: white !important;
        }

        /* Status badges */
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 500;
            font-size: 0.9rem;
        }

        /* Progress bars */
        .progress-bar {
            background: var(--border-color);
            border-radius: 4px;
            height: 6px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        /* Form inputs */
        .stTextInput>div>div>input {
            background: var(--bg-light);
            color: var(--text-color);
            border: 2px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem;
        }

        .stTextInput>div>div>input:focus {
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }

        /* Table styles */
        .table-container {
            background: var(--bg-card);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }

        .table-row {
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
            transition: background-color 0.2s ease;
        }

        .table-row:hover {
            background: var(--bg-light);
        }

        /* Image container */
        .image-container {
            background: var(--bg-card);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }

        .image-container img {
            border-radius: 8px;
        }

        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
        }

        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            color: var(--text-color);
            border-radius: 6px;
            font-size: 0.85rem;
            white-space: nowrap;
            z-index: 1000;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow);
        }

        /* Messages */
        .success-message, .error-message, .info-message, .warning-message {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid transparent;
        }

        .success-message {
            background: rgba(46, 204, 113, 0.1);
            border-color: var(--success-color);
            color: var(--success-color);
        }

        .error-message {
            background: rgba(231, 76, 60, 0.1);
            border-color: var(--danger-color);
            color: var(--danger-color);
        }

        .warning-message {
            background: rgba(241, 196, 15, 0.1);
            border-color: var(--warning-color);
            color: var(--warning-color);
        }

        .info-message {
            background: rgba(52, 152, 219, 0.1);
            border-color: var(--secondary-color);
            color: var(--secondary-color);
        }

        /* Dark mode specific overrides */
        @media (prefers-color-scheme: dark) {
            .card {
                background: var(--bg-card);
            }
            
            .stTextInput>div>div>input {
                background: var(--bg-light);
            }
            
            .progress-bar {
                background: rgba(255, 255, 255, 0.1);
            }
            
            .tooltip:hover::after {
                background: var(--bg-light);
            }
            
            .success-message, .error-message, .info-message, .warning-message {
                background: rgba(255, 255, 255, 0.05);
            }
        }

        /* File uploader styling */
        .stFileUploader > div {
            padding: 2rem;
            border: 2px dashed var(--secondary-color);
            border-radius: 12px;
            background: var(--bg-light);
            transition: all 0.3s ease;
            margin: 2rem auto;
            max-width: 800px;
        }
        
        .stFileUploader > div:hover {
            border-color: var(--primary-color);
            background: var(--bg-card);
            transform: translateY(-2px);
        }
        
        .stFileUploader [data-testid="stFileUploadDropzone"] {
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
        }
        
        .stFileUploader [data-testid="stMarkdownContainer"] p {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .upload-section {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .processing-queue {
            margin-top: 2rem;
            padding: 1rem;
            border-radius: 8px;
            background: var(--bg-card);
        }
        
        .queue-item {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .queue-item:last-child {
            border-bottom: none;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description with modern styling
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1>JSW Engineering Drawing DataSheet Extractor</h1>
            <div style="color: var(--text-muted); font-size: 1.1rem; margin: 1rem 0;">
                Automatically extract and analyze technical specifications from engineering drawings
            </div>
            <div class="progress-bar" style="max-width: 200px; margin: 2rem auto;">
                <div class="progress-bar-fill" style="width: 100%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

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
    if 'current_image' not in st.session_state:
        st.session_state.current_image = {}
    if 'edited_values' not in st.session_state:
        st.session_state.edited_values = {}

    # File uploader with modern styling
    if st.session_state.selected_drawing is None:
        # Initialize processing queue if not exists
        if 'processing_queue' not in st.session_state:
            st.session_state.processing_queue = []
        if 'currently_processing' not in st.session_state:
            st.session_state.currently_processing = False

        # Custom styling for file uploader
        st.markdown("""
            <style>
            .stFileUploader > div {
                padding: 2rem;
                border: 2px dashed var(--secondary-color);
                border-radius: 12px;
                background: var(--bg-light);
                transition: all 0.3s ease;
                margin: 2rem auto;
                max-width: 800px;
            }
            
            .stFileUploader > div:hover {
                border-color: var(--primary-color);
                background: var(--bg-card);
                transform: translateY(-2px);
            }
            
            .stFileUploader [data-testid="stFileUploadDropzone"] {
                min-height: 200px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--text-muted);
            }
            
            .stFileUploader [data-testid="stMarkdownContainer"] p {
                color: var(--text-muted);
                font-size: 0.9rem;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Multi-file uploader without header
        uploaded_files = st.file_uploader("", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

        if uploaded_files:
            # Add new files to processing queue
            for uploaded_file in uploaded_files:
                if uploaded_file not in st.session_state.processing_queue:
                    st.session_state.processing_queue.append(uploaded_file)

            # Display uploaded files and processing options
            st.markdown("""
                <div class="card" style="margin-top: 2rem;">
                    <h4 style="color: var(--primary-color); margin-bottom: 1rem;">Uploaded Drawings</h4>
                </div>
            """, unsafe_allow_html=True)

            # Process each uploaded file
            for idx, file in enumerate(uploaded_files):
                with st.container():
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Show preview of the image
                        st.image(file, caption=f"Preview: {file.name}", use_column_width=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div class="card" style="padding: 1rem;">
                                <strong style="color: var(--primary-color);">{file.name}</strong>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Add process button
                        if st.button(f"Process Drawing", key=f"process_{idx}"):
                            try:
                                # Process the file
                                file.seek(0)
                                image_bytes = file.read()
                                
                                # Step 1: Identify drawing type
                                with st.spinner('Identifying drawing type...'):
                                    drawing_type = identify_drawing_type(image_bytes)
                                    
                                    if not drawing_type or "❌" in drawing_type:
                                        st.error(drawing_type if drawing_type else "❌ Could not identify drawing type")
                                        continue
                                    
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
                                    
                                    # Process the drawing based on type
                                    with st.spinner(f'Analyzing {drawing_type.lower()} drawing...'):
                                        result = None
                                        if drawing_type == "CYLINDER":
                                            result = analyze_cylinder_image(image_bytes)
                                        elif drawing_type == "VALVE":
                                            result = analyze_valve_image(image_bytes)
                                        elif drawing_type == "GEARBOX":
                                            result = analyze_gearbox_image(image_bytes)
                                        elif drawing_type == "NUT":
                                            result = analyze_nut_image(image_bytes)
                                        elif drawing_type == "LIFTING RAM":
                                            result = analyze_lifting_ram_image(image_bytes)
                                        
                                        if result and "❌" not in result:
                                            # Update with successful results
                                            parsed_results = parse_ai_response(result)
                                            drawing_number = (parsed_results.get('MODEL NO', '') 
                                                            if drawing_type == "VALVE" 
                                                            else parsed_results.get('DRAWING NUMBER', ''))
                                            
                                            if not drawing_number or drawing_number == 'Unknown':
                                                drawing_number = f"{drawing_type}_{len(st.session_state.drawings_table)}"
                                            
                                            # Store the image
                                            file.seek(0)
                                            st.session_state.current_image[drawing_number] = file.read()
                                            st.session_state.all_results[drawing_number] = parsed_results
                                            
                                            # Update status
                                            parameters = get_parameters_for_type(drawing_type)
                                            non_empty_fields = sum(1 for k in parameters if parsed_results.get(k, '').strip())
                                            total_fields = len(parameters)
                                            
                                            new_drawing.update({
                                                'Drawing No.': drawing_number,
                                                'Processing Status': 'Completed' if non_empty_fields == total_fields else 'Needs Review!',
                                                'Extracted Fields Count': f"{non_empty_fields}/{total_fields}",
                                                'Confidence Score': f"{(non_empty_fields / total_fields * 100):.0f}%"
                                            })
                                            
                                            st.success(f"✅ Successfully processed {file.name}")
                                        else:
                                            st.error(f"❌ Failed to process {file.name}")
                                            new_drawing.update({
                                                'Processing Status': 'Failed',
                                                'Confidence Score': '0%',
                                                'Extracted Fields Count': '0/0'
                                            })
                                        
                                        # Update the table
                                        st.session_state.drawings_table.iloc[-1] = new_drawing
                                        
                            except Exception as e:
                                st.error(f"❌ Error processing {file.name}: {str(e)}")
                            
                            st.experimental_rerun()
                    
                    st.markdown("<hr>", unsafe_allow_html=True)

    # Display the drawings table with improved styling
    if not st.session_state.drawings_table.empty:
        st.markdown("""
            <div class="card">
                <h3>Processed Drawings</h3>
                <div style="color: var(--text-muted); margin-bottom: 1.5rem;">
                    View and manage your processed technical drawings
                </div>
                <div class="table-container">
            """, unsafe_allow_html=True)
        
        # Create a clean table layout
        for index, row in st.session_state.drawings_table.iterrows():
            with st.container():
                st.markdown(f"""
                    <div class="card" style="margin-bottom: 1rem; transition: all 0.2s ease;">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 1])
                
                with col1:
                    st.markdown(f"""
                        <div class="tooltip" data-tooltip="Drawing Type">
                            <strong style="color: var(--primary-color);">{row['Drawing Type']}</strong>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="tooltip" data-tooltip="Drawing Number">
                            {row['Drawing No.']}
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    status_styles = {
                        'Processing..': ('var(--secondary-color)', 'rgba(52, 152, 219, 0.1)'),
                        'Completed': ('var(--success-color)', 'rgba(39, 174, 96, 0.1)'),
                        'Needs Review!': ('var(--warning-color)', 'rgba(243, 156, 18, 0.1)'),
                        'Failed': ('var(--danger-color)', 'rgba(231, 76, 60, 0.1)')
                    }
                    color, bg = status_styles.get(row['Processing Status'], ('black', 'rgba(0, 0, 0, 0.1)'))
                    st.markdown(f"""
                        <div class="status-badge" style="background: {bg}; color: {color};">
                            {row['Processing Status']}
                        </div>
                    """, unsafe_allow_html=True)
                with col4:
                    fields = row['Extracted Fields Count'].split('/')
                    percentage = (int(fields[0]) / int(fields[1])) * 100 if fields[1] != '0' else 0
                    st.markdown(f"""
                        <div class="tooltip" data-tooltip="Extracted Fields">
                            <div style="margin-bottom: 0.25rem;">{row['Extracted Fields Count']}</div>
                            <div class="progress-bar" style="height: 4px;">
                                <div class="progress-bar-fill" style="width: {percentage}%;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                with col5:
                    confidence = int(row['Confidence Score'].rstrip('%'))
                    color = '#27AE60' if confidence >= 80 else '#F39C12' if confidence >= 50 else '#E74C3C'
                    st.markdown(f"""
                        <div class="tooltip" data-tooltip="Confidence Score">
                            <div style="margin-bottom: 0.25rem;">{row['Confidence Score']}</div>
                            <div class="progress-bar" style="height: 4px;">
                                <div class="progress-bar-fill" style="width: {confidence}%; background: {color};"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                with col6:
                    if st.button('View', key=f'view_{index}'):
                        st.session_state.selected_drawing = row['Drawing No.']
                        st.experimental_rerun()
                
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Detailed view with improved styling
    if st.session_state.selected_drawing and st.session_state.selected_drawing in st.session_state.all_results:
        st.markdown(f"""
            <div class="card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
                    <div>
                        <h3 style="margin: 0;">Detailed View: {st.session_state.selected_drawing}</h3>
                        <div style="color: var(--text-muted);">
                            Review and edit extracted specifications
                        </div>
                    </div>
                    <div class="tooltip" data-tooltip="Return to drawings list">
                        <button class="back-button" onclick="window.history.back()">
                            <i class="fas fa-arrow-left"></i> Back
                        </button>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create two columns with better spacing
        image_col, edit_col = st.columns([1, 2])

        with image_col:
            st.markdown("""
                <div class="card image-container">
            """, unsafe_allow_html=True)
            
            image_data = st.session_state.current_image.get(st.session_state.selected_drawing)
            if image_data is not None:
                try:
                    image = Image.open(io.BytesIO(image_data))
                    st.image(image, caption="Technical Drawing", use_column_width=True)
                except Exception as e:
                    st.error("Unable to display image. Please try processing the drawing again.")
            else:
                st.warning("Image not available. Please try processing the drawing again.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with edit_col:
            st.markdown("""
                <div class="card">
                    <h4 style="margin-bottom: 1.5rem;">Edit Specifications</h4>
            """, unsafe_allow_html=True)
            
            results = st.session_state.all_results[st.session_state.selected_drawing]
            drawing_type = st.session_state.drawings_table[
                st.session_state.drawings_table['Drawing No.'] == st.session_state.selected_drawing
            ]['Drawing Type'].iloc[0]
            
            # Initialize edited values for this drawing if not exists
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
                    # Set specific confidence scores for CLOSE LENGTH and STROKE LENGTH
                    if param == "CLOSE LENGTH" and current_value.strip():
                        confidence = "80%"
                    elif param == "STROKE LENGTH" and current_value.strip():
                        confidence = "90%"
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

if __name__ == "__main__":
    main()
