import streamlit as st
import base64
from PIL import Image
import io
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import sys
import subprocess
import tempfile

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="JSW Engineering Drawing DataSheet Extractor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load API key
load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    st.error("❌ API key not found! Check your .env file.")
    st.stop()

# Check for required packages
def check_dependencies():
    try:
        import pdf2image
        return True
    except ImportError:
        st.error("""
        ❌ Required package 'pdf2image' is not installed. Please install the following:
        
        1. Install pdf2image:
        ```bash
        pip install pdf2image
        ```
        
        2. Install poppler:
        - On Ubuntu/Debian:
        ```bash
        apt-get install poppler-utils
        ```
        - On macOS:
        ```bash
        brew install poppler
        ```
        - On Windows:
        Download and install poppler from: http://blog.alivate.com.au/poppler-windows/
        
        After installing, please restart the application.
        """)
        return False

# Only import pdf2image if dependencies are available
if check_dependencies():
    from pdf2image import convert_from_bytes

def handle_pdf_upload(pdf_bytes):
    """Handle PDF upload with better error handling"""
    if not check_dependencies():
        return None
        
    try:
        with st.spinner("Converting PDF to images..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf.seek(0)
                try:
                    images = convert_from_bytes(pdf_bytes)
                    if images:
                        st.success(f"✅ PDF converted successfully! {len(images)} pages found.")
                        return images
                    else:
                        st.error("❌ No pages found in PDF.")
                        return None
                except Exception as e:
                    st.error(f"❌ Error converting PDF: {str(e)}")
                    st.info("Please make sure poppler is installed correctly.")
                    return None
    except Exception as e:
        st.error(f"❌ Error handling PDF: {str(e)}")
        return None
    finally:
        if 'temp_pdf' in locals():
            try:
                os.unlink(temp_pdf.name)
            except:
                pass

def pil_to_bytes(pil_image):
    """Convert PIL Image to bytes"""
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

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
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response_json = response.json()
        
        if response.status_code == 200 and "choices" in response_json:
            drawing_type = response_json["choices"][0]["message"]["content"].strip().upper()
            # Clean up the response to handle potential duplicates
            if "CYLINDER" in drawing_type:
                return "CYLINDER"
            elif "VALVE" in drawing_type:
                return "VALVE"
            elif "GEARBOX" in drawing_type:
                return "GEARBOX"
            else:
                return f"❌ Invalid drawing type: {drawing_type}"
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
        .pdf-page-button {
            margin: 5px;
            padding: 5px 10px;
        }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            margin: 1rem 0;
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
    if 'pdf_images' not in st.session_state:
        st.session_state.pdf_images = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

    # File uploader and processing section
    if st.session_state.selected_drawing is None:
        uploaded_file = st.file_uploader("Select File", type=['png', 'jpg', 'jpeg', 'pdf'])

        if uploaded_file is not None:
            # Handle PDF files
            if uploaded_file.type == "application/pdf":
                if st.session_state.pdf_images is None:
                    pdf_bytes = uploaded_file.read()
                    st.session_state.pdf_images = handle_pdf_upload(pdf_bytes)
                    if not st.session_state.pdf_images:
                        return

                # Show PDF navigation
                if st.session_state.pdf_images:
                    st.write("### Select Page to Process")
                    cols = st.columns(min(5, len(st.session_state.pdf_images)))
                    for i, col in enumerate(cols):
                        if i < len(st.session_state.pdf_images):
                            if col.button(f"Page {i+1}", key=f"page_{i}", help=f"View page {i+1}"):
                                st.session_state.current_page = i

                    # Display current page
                    current_image = st.session_state.pdf_images[st.session_state.current_page]
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        if st.button("Process Current Page", key="process_button"):
                            try:
                                image_bytes = pil_to_bytes(current_image)
                                process_image(image_bytes, st)
                            except Exception as e:
                                st.error(f"❌ An error occurred: {str(e)}")
                    
                    with col2:
                        st.image(current_image, caption=f"Page {st.session_state.current_page + 1}", use_column_width=True)

            else:  # Handle image files
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    if st.button("Process Drawing", key="process_button"):
                        try:
                            uploaded_file.seek(0)
                            image_bytes = uploaded_file.read()
                            process_image(image_bytes, st)
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
        
        # Create detailed parameters table
        parameters = get_parameters_for_type(drawing_type)
        detailed_data = []
        
        for param in parameters:
            value = results.get(param, '')
            confidence = "100%" if value.strip() else "0%"
            action = "✅ Auto-filled" if value.strip() else "🔴 Manual Detection Required"
            
            detailed_data.append({
                "Parameter": param,
                "Value": value if value.strip() else "Not detected",
                "Confidence": confidence,
                "Action": action
            })
        
        # Display detailed data
        detailed_df = pd.DataFrame(detailed_data)
        st.table(detailed_df)
        
        # Add export and back buttons
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

def process_image(image_bytes, st):
    """Process a single image"""
    # Step 1: Identify drawing type
    with st.spinner('Identifying drawing type...'):
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

if __name__ == "__main__":
    main()
