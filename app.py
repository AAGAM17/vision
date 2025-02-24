import streamlit as st
import base64
from PIL import Image
import io
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import datetime

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

    headers = {
        "Authorization": f"Bearer {st.session_state.current_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = process_api_response(response, analyze_cylinder_image, image_bytes)
        if "❌" not in result:
            # Parse results
            parsed_results = parse_ai_response(result)
            
            # Clean up mounting and rod end values
            if parsed_results.get('MOUNTING', '').strip() in ['[value]', '[Value]', '[VALUES]']:
                parsed_results['MOUNTING'] = ''
            if parsed_results.get('ROD END', '').strip() in ['[value]', '[Value]', '[VALUES]']:
                parsed_results['ROD END'] = ''
            
            # Convert HLP to HYD. OIL MINERAL if needed
            if parsed_results.get('FLUID', '').strip().upper() == 'HLP':
                parsed_results['FLUID'] = 'HYD. OIL MINERAL'
            
            # Process temperature range to get maximum value
            temp = parsed_results.get('OPERATING TEMPERATURE', '').strip()
            if temp:
                # Handle different range formats
                if 'TO' in temp.upper():
                    # Format: "40 TO 50 DEG C"
                    max_temp = temp.upper().split('TO')[-1].split('DEG')[0].strip()
                elif '+' in temp:
                    # Format: "-10°C +60°C" or similar
                    max_temp = temp.split('+')[-1].split('DEG')[0].strip()
                else:
                    # Single value or other format
                    max_temp = temp.split('DEG')[0].strip()
                
                # Clean up the max temperature value
                max_temp = ''.join(filter(lambda x: x.isdigit() or x == '.', max_temp))
                if max_temp:
                    parsed_results['OPERATING TEMPERATURE'] = f"{max_temp} DEG C"
                
            return '\n'.join([f"{k}: {v}" for k, v in parsed_results.items()])
        return result
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
                            "Analyze the gearbox engineering drawing and extract ONLY the values marked in RED plus the drawing number.\n"
                            "STRICT RULES:\n"
                            "1) If a value is missing or unclear, return an empty string. DO NOT estimate any values.\n"
                            "2) Extract and return data in this format:\n"
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
    """Identify if the drawing is a cylinder, valve, gearbox, hex nut, or lifting ram"""
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
                            "4. Hex Nut\n"
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
            elif "NUT" in drawing_type:
                return "NUT"
            elif "LIFTING_RAM" in drawing_type or "LIFTING RAM" in drawing_type:
                return "LIFTING_RAM"
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
                            "Analyze the nut drawing and extract ONLY the values marked in RED plus the drawing number.\n"
                            "STRICT RULES:\n"
                            "1) If a value is missing or unclear, return an empty string. DO NOT estimate any values.\n"
                            "2) Extract and return data in this format:\n"
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

    headers = {
        "Authorization": f"Bearer {st.session_state.current_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return process_api_response(response, analyze_nut_image, image_bytes)
    except Exception as e:
        return f"❌ Processing Error: {str(e)}"

def analyze_lifting_ram_image(image_bytes):
    """Analyze lifting ram drawings and extract technical specifications"""
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
                            "Analyze this single-stage lifting ram technical data and extract the specifications.\n"
                            "STRICT RULES:\n"
                            "1) If a value is missing or unclear, return an empty string. DO NOT estimate any values.\n"
                            "2) Extract and return data in this EXACT format with units:\n"
                            "HEIGHT: [value] mm\n"
                            "TOTAL STROKE: [value] mm\n"
                            "PISTON STROKE: [value] mm\n"
                            "PISTON LIFTING FORCE: [value] kN\n"
                            "WEIGHT: [value] kg\n"
                            "OIL VOLUME: [value] l\n"
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
        return process_api_response(response, analyze_lifting_ram_image, image_bytes)
    except Exception as e:
        return f"❌ Processing Error: {str(e)}"

def submit_feedback_to_company(feedback_data, drawing_info, additional_notes=""):
    """
    Submit feedback to the company's system
    Returns: (success: bool, message: str)
    """
    try:
        # Create a comprehensive feedback package
        feedback_package = {
            "timestamp": datetime.datetime.now().isoformat(),
            "drawing_info": {
                "drawing_number": drawing_info.get("drawing_number", ""),
                "drawing_type": drawing_info.get("drawing_type", ""),
                "processing_date": datetime.datetime.now().strftime("%Y-%m-%d")
            },
            "corrections": feedback_data,
            "additional_notes": additional_notes,
            "user_info": {
                "session_id": st.session_state.get("session_id", "unknown"),
                "submission_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Here you would implement the actual API call to your company's feedback system
        # For now, we'll just log it and store in session state
        st.session_state.feedback_history.append(feedback_package)
        
        # In a real implementation, you would send this to your backend:
        # response = requests.post(
        #     "https://your-company-api.com/feedback",
        #     json=feedback_package,
        #     headers={"Authorization": "Bearer " + API_KEY}
        # )
        # if response.status_code != 200:
        #     return False, "Failed to submit feedback to server"
        
        return True, "Feedback submitted successfully"
    except Exception as e:
        return False, f"Error submitting feedback: {str(e)}"

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
            background: linear-gradient(135deg, var(--secondary-color), #2980B9) !important;
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
            background: linear-gradient(135deg, #2980B9, #2573a7) !important;
            opacity: 0.9;
        }

        .stButton>button:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        /* Primary button */
        .stButton.primary>button,
        button[data-baseweb="button"].primary {
            background: linear-gradient(135deg, #3498DB, #2980B9) !important;
            color: white !important;
        }

        /* Secondary button */
        .stButton.secondary>button,
        button[data-baseweb="button"].secondary {
            background: linear-gradient(135deg, #7F8C8D, #34495E) !important;
            color: white !important;
        }

        /* Success button */
        .stButton.success>button,
        button[data-baseweb="button"].success {
            background: linear-gradient(135deg, #2ECC71, #27AE60) !important;
            color: white !important;
        }

        /* Warning button */
        .stButton.warning>button,
        button[data-baseweb="button"].warning {
            background: linear-gradient(135deg, #F1C40F, #F39C12) !important;
            color: white !important;
        }

        /* Danger button */
        .stButton.danger>button,
        button[data-baseweb="button"].danger {
            background: linear-gradient(135deg, #E74C3C, #C0392B) !important;
            color: white !important;
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
            background: linear-gradient(135deg, #7F8C8D, #34495E) !important;
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
            background: linear-gradient(135deg, #34495E, #2C3E50) !important;
        }

        /* Make sure text in buttons is always white */
        .stButton>button>div,
        .stButton>button>div>p,
        .stButton>button>div>div,
        .stDownloadButton>button>div {
            color: white !important;
        }

        /* Ensure button text remains visible in both modes */
        [data-theme="light"] .stButton>button,
        [data-theme="dark"] .stButton>button,
        [data-theme="light"] .stDownloadButton>button,
        [data-theme="dark"] .stDownloadButton>button {
            color: white !important;
        }

        [data-theme="light"] .stButton>button>div,
        [data-theme="dark"] .stButton>button>div,
        [data-theme="light"] .stDownloadButton>button>div,
        [data-theme="dark"] .stDownloadButton>button>div {
            color: white !important;
        }

        /* Additional button visibility fixes */
        .stButton>button[kind="secondary"],
        .stDownloadButton>button[kind="secondary"] {
            background: linear-gradient(135deg, #7F8C8D, #34495E) !important;
        }

        .stButton>button[kind="primary"],
        .stDownloadButton>button[kind="primary"] {
            background: linear-gradient(135deg, #3498DB, #2980B9) !important;
        }

        /* Button container styling */
        .button-container {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
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
            padding: 1rem;
            border: 2px dashed var(--secondary-color);
            border-radius: 8px;
            background: var(--bg-light);
            transition: all 0.3s ease;
            margin: 1rem auto;
            max-width: 600px;
        }
        
        .stFileUploader > div:hover {
            border-color: var(--primary-color);
            background: var(--bg-card);
            transform: translateY(-1px);
        }
        
        .stFileUploader [data-testid="stFileUploadDropzone"] {
            min-height: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
        }
        
        .stFileUploader [data-testid="stMarkdownContainer"] p {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        /* Compact uploaded drawings section */
        .uploaded-drawing {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }

        .uploaded-drawing img {
            max-width: 150px;
            height: auto;
            border-radius: 4px;
        }

        .drawing-info {
            flex-grow: 1;
        }

        .drawing-actions {
            display: flex;
            gap: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description with modern styling
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1>JSW Engineering Drawing DataSheet Extractor</h1>
            <div style="color: var(--text-muted); font-size: 1.1rem; margin: 0.5rem 0;">
                Automatically extract and analyze technical specifications from engineering drawings
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
    if 'custom_products' not in st.session_state:
        st.session_state.custom_products = {}
    if 'show_feedback_popup' not in st.session_state:
        st.session_state.show_feedback_popup = False
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = {}
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []
    if 'feedback_status' not in st.session_state:
        st.session_state.feedback_status = None

    # Add New Product Section
    with st.sidebar:
        st.markdown("### Add New Product Type")
        new_product_name = st.text_input("Product Name (e.g., BEARING, SHAFT)", key="new_product_name")
        
        # Dynamic parameter addition
        st.markdown("#### Required Parameters")
        param_name = st.text_input("Parameter Name", key="param_name")
        param_unit = st.text_input("Unit (optional)", key="param_unit")
        
        if st.button("Add Parameter"):
            if param_name:
                if new_product_name not in st.session_state.custom_products:
                    st.session_state.custom_products[new_product_name] = []
                param_with_unit = f"{param_name} [{param_unit}]" if param_unit else param_name
                st.session_state.custom_products[new_product_name].append(param_with_unit)
                st.success(f"Added parameter: {param_with_unit}")
        
        # Show current parameters
        if new_product_name in st.session_state.custom_products:
            st.markdown("#### Current Parameters:")
            for param in st.session_state.custom_products[new_product_name]:
                st.write(f"- {param}")
        
        if st.button("Save New Product"):
            if new_product_name and new_product_name in st.session_state.custom_products:
                # Add to parameters list
                parameters = st.session_state.custom_products[new_product_name]
                parameters.append("DRAWING NUMBER")  # Always include drawing number
                
                # Create analysis function name
                func_name = f"analyze_{new_product_name.lower()}_image"
                
                # Create prompt template
                prompt = (
                    f"Analyze this {new_product_name.lower()} drawing and extract the following parameters.\n"
                    "STRICT RULES:\n"
                    "1) If a value is missing or unclear, return an empty string. DO NOT estimate any values.\n"
                    "2) Extract and return data in this format:\n"
                )
                for param in parameters:
                    prompt += f"{param}: [value]\n"
                
                st.session_state.custom_products[new_product_name] = {
                    'parameters': parameters,
                    'prompt': prompt
                }
                st.success(f"✅ Successfully added {new_product_name} to the system!")
                
    # File uploader with modern styling
    if st.session_state.selected_drawing is None:
        # Initialize processing queue if not exists
        if 'processing_queue' not in st.session_state:
            st.session_state.processing_queue = []
        if 'currently_processing' not in st.session_state:
            st.session_state.currently_processing = False

        # Multi-file uploader without header
        uploaded_files = st.file_uploader("", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

        if uploaded_files:
            # Add new files to processing queue
            for uploaded_file in uploaded_files:
                if uploaded_file not in st.session_state.processing_queue:
                    st.session_state.processing_queue.append(uploaded_file)

            # Display uploaded files in a compact layout
            st.markdown("""
                <div class="card" style="margin-top: 1rem; padding: 1rem;">
                    <h4 style="color: var(--primary-color); margin-bottom: 0.5rem;">Uploaded Drawings</h4>
                    <div class="uploaded-drawings-container">
            """, unsafe_allow_html=True)

            # Process each uploaded file
            for idx, file in enumerate(uploaded_files):
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    # Show smaller preview of the image
                    st.image(file, width=150)
                
                with col2:
                    # Show file info and processing status
                    st.markdown(f"""
                        <div style="margin-bottom: 0.5rem;">
                            <strong style="color: var(--primary-color);">{file.name}</strong>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col2_1, col2_2 = st.columns([3, 2])
                    
                    with col2_1:
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
                                        elif drawing_type == "LIFTING_RAM":
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
                                            
                                            # Add view button after successful processing
                                            st.markdown(f"""
                                                <div style="margin-top: 0.5rem;">
                                                    <div class="status-badge" style="background: rgba(39, 174, 96, 0.1); color: var(--success-color);">
                                                        Processed Successfully
                                                    </div>
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                                            if st.button("View Results", key=f"view_results_{idx}"):
                                                st.session_state.selected_drawing = drawing_number
                                                st.experimental_rerun()
                                            
                                        else:
                                            st.error(f"❌ Failed to process {file.name}")
                                            new_drawing.update({
                                                'Processing Status': 'Failed',
                                                'Confidence Score': '0%',
                                                'Extracted Fields Count': '0/0'
                                            })
                                            
                                            # Show error status
                                            st.markdown(f"""
                                                <div style="margin-top: 0.5rem;">
                                                    <div class="status-badge" style="background: rgba(231, 76, 60, 0.1); color: var(--danger-color);">
                                                        Processing Failed
                                                    </div>
                                                </div>
                                            """, unsafe_allow_html=True)
                                        
                                        # Update the table
                                        st.session_state.drawings_table.iloc[-1] = new_drawing
                            except Exception as e:
                                st.error(f"❌ Error processing {file.name}: {str(e)}")
                            st.experimental_rerun()
                    
                    with col2_2:
                        # Show processing status if already processed
                        if not st.session_state.drawings_table.empty:
                            matching_drawing = st.session_state.drawings_table[
                                st.session_state.drawings_table['Drawing No.'].str.contains(file.name, na=False)
                            ]
                            if not matching_drawing.empty:
                                status = matching_drawing.iloc[0]['Processing Status']
                                status_styles = {
                                    'Processing..': ('var(--secondary-color)', 'rgba(52, 152, 219, 0.1)'),
                                    'Completed': ('var(--success-color)', 'rgba(39, 174, 96, 0.1)'),
                                    'Needs Review!': ('var(--warning-color)', 'rgba(243, 156, 18, 0.1)'),
                                    'Failed': ('var(--danger-color)', 'rgba(231, 76, 60, 0.1)')
                                }
                                color, bg = status_styles.get(status, ('black', 'rgba(0, 0, 0, 0.1)'))
                                st.markdown(f"""
                                    <div class="status-badge" style="background: {bg}; color: {color};">
                                        {status}
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                if status in ['Completed', 'Needs Review!']:
                                    drawing_number = matching_drawing.iloc[0]['Drawing No.']
                                    if st.button("View Results", key=f"view_{drawing_number}"):
                                        st.session_state.selected_drawing = drawing_number
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
                <div style="margin-bottom: 1.5rem;">
                    <h3 style="margin: 0;">Detailed View: {st.session_state.selected_drawing}</h3>
                    <div style="color: var(--text-muted);">
                        Review and edit extracted specifications
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
            st.markdown("""
                <div style="display: flex; gap: 1rem; margin-top: 2rem;">
                    <div style="flex: 1;">
                        <div class="button-container">
            """, unsafe_allow_html=True)
            
            # Create a 2x2 grid for buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Back to All Drawings", type="secondary", use_container_width=True):
                    st.session_state.selected_drawing = None
                    st.experimental_rerun()
                
                # Create DataFrame for export
                export_df = pd.DataFrame(edited_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Export to CSV",
                    data=csv,
                    file_name=f"{st.session_state.selected_drawing}_details.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

            with col2:
                if st.button("Save Changes", type="primary", use_container_width=True):
                    # Collect changes for feedback
                    feedback_data = {}
                    for param, value in st.session_state.edited_values[st.session_state.selected_drawing].items():
                        if value.strip() and value != results.get(param, ''):
                            feedback_data[param] = {
                                'original': results.get(param, ''),
                                'corrected': value
                            }
                    
                    # Update the results
                    for param, value in st.session_state.edited_values[st.session_state.selected_drawing].items():
                        if value.strip():  # Only update non-empty values
                            results[param] = value
                    st.session_state.all_results[st.session_state.selected_drawing] = results
                    
                    # If there are changes, show feedback popup
                    if feedback_data:
                        st.session_state.feedback_data = feedback_data
                        st.session_state.show_feedback_popup = True
                    
                    st.success("✅ Changes saved successfully!")
                
                # Create a clean format of the table with all columns
                values_text = []
                for row in edited_data:
                    param = row['Parameter']
                    value = row['Value'].strip() if row['Value'] else ""
                    values_text.append(f"{param}\t{value}")
                
                # Add Copy Values button with completely hidden implementation
                st.markdown("""
                    <div style="display:none">
                        <script>
                        function copyToClipboard(text) {
                            const textarea = document.createElement('textarea');
                            textarea.value = text;
                            document.body.appendChild(textarea);
                            textarea.select();
                            try {
                                document.execCommand('copy');
                                const button = document.getElementById('copyButton');
                                button.innerHTML = '✓ Copied!';
                                setTimeout(() => button.innerHTML = 'Copy Values', 2000);
                            } catch (err) {
                                console.error('Failed to copy:', err);
                            }
                            document.body.removeChild(textarea);
                        }
                        </script>
                    </div>
                    <button
                        id="copyButton"
                        onclick="copyToClipboard(`{('\\n'.join(values_text)).replace('"', '\\"').replace("'", "\\'")}`)"
                        style="
                            background: linear-gradient(135deg, #3498DB, #2980B9);
                            color: white;
                            border: none;
                            padding: 0.75rem 1.5rem;
                            border-radius: 8px;
                            font-weight: 600;
                            cursor: pointer;
                            width: 100%;
                            height: 44px;
                            transition: all 0.3s ease;
                        "
                    >
                        Copy Values
                    </button>
                """, unsafe_allow_html=True)

            st.markdown("""
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Feedback Popup
    if st.session_state.show_feedback_popup:
        st.markdown("""
            <div class="card" style="padding: 2rem; max-width: 800px; margin: 2rem auto;">
                <h3 style="margin-bottom: 1.5rem;">Submit Feedback</h3>
                <div style="color: var(--text-muted); margin-bottom: 2rem;">
                    Your corrections will help improve our extraction system
                </div>
            </div>
        """, unsafe_allow_html=True)

        with st.form("feedback_form", clear_on_submit=True):
            # Display corrections in a table format
            st.markdown("#### Changes Detected")
            for param, values in st.session_state.feedback_data.items():
                st.markdown("""
                    <div style="background: var(--bg-light); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                        <div style="font-weight: bold; color: var(--primary-color); margin-bottom: 0.5rem;">
                            {param}
                        </div>
                        <div style="display: flex; gap: 2rem;">
                            <div>
                                <span style="color: var(--text-muted);">Original:</span>
                                <span style="color: var(--danger-color);">{values['original'] or '(empty)'}</span>
                            </div>
                            <div>
                                <span style="color: var(--text-muted);">Corrected:</span>
                                <span style="color: var(--success-color);">{values['corrected']}</span>
                            </div>
                        </div>
                    </div>
                """.format(param=param, values=values), unsafe_allow_html=True)
            
            # Additional feedback options
            st.markdown("#### Additional Information")
            feedback_category = st.selectbox(
                "Feedback Category",
                ["Value Correction", "Missing Information", "Wrong Recognition", "Other"]
            )
            
            additional_notes = st.text_area(
                "Additional Notes (optional)",
                placeholder="Please provide any additional context or observations..."
            )
            
            # Submission buttons
            col1, col2 = st.columns(2)
            with col1:
                submit_button = st.form_submit_button(
                    "Submit Feedback",
                    type="primary",
                    use_container_width=True
                )
            with col2:
                cancel_button = st.form_submit_button(
                    "Cancel",
                    type="secondary",
                    use_container_width=True
                )
            
            if submit_button:
                # Get current drawing info
                drawing_info = {
                    "drawing_number": st.session_state.selected_drawing,
                    "drawing_type": st.session_state.drawings_table[
                        st.session_state.drawings_table['Drawing No.'] == st.session_state.selected_drawing
                    ]['Drawing Type'].iloc[0]
                }
                
                # Add category to feedback data
                feedback_data = {
                    "corrections": st.session_state.feedback_data,
                    "category": feedback_category,
                    "notes": additional_notes
                }
                
                # Submit feedback
                success, message = submit_feedback_to_company(
                    feedback_data,
                    drawing_info,
                    additional_notes
                )
                
                if success:
                    st.session_state.feedback_status = {
                        "type": "success",
                        "message": "✅ " + message
                    }
                    # Clear feedback popup
                    st.session_state.show_feedback_popup = False
                    st.session_state.feedback_data = {}
                else:
                    st.session_state.feedback_status = {
                        "type": "error",
                        "message": "❌ " + message
                    }
                
                st.experimental_rerun()
            
            elif cancel_button:
                st.session_state.show_feedback_popup = False
                st.session_state.feedback_data = {}
                st.experimental_rerun()

    # Display feedback status if exists
    if st.session_state.feedback_status:
        status_type = st.session_state.feedback_status["type"]
        message = st.session_state.feedback_status["message"]
        
        if status_type == "success":
            st.success(message)
        else:
            st.error(message)
        
        # Clear status after displaying
        st.session_state.feedback_status = None

if __name__ == "__main__":
    main()
