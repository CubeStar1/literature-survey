import streamlit as st
import openai
import google.generativeai as genai
import PyPDF2
import tempfile
import os
from pathlib import Path
import time
from datetime import datetime
import json
import re


def setup_output_directory():
    """Create output directory for summaries"""
    output_dir = Path("paper_summaries") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_summary_filename(pdf_filename):
    """Generate summary filename based on PDF filename"""
    base_name = Path(pdf_filename).stem
    return f"{base_name}_summary.txt"


def save_summary(summary, filename, output_dir):
    """Save summary to file and return the path"""
    output_path = output_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    return output_path


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def create_markdown_table_header():
    """Create the header for the markdown table"""
    return "| Paper Name | Authors | Year | Publication | Key Findings |\n|------------|----------|------|---------|---------------|\n"


def add_to_markdown_table(paper_info):
    """Add a paper's information to the markdown table format"""
    # Escape any pipe characters in the text fields
    name = paper_info['title'].replace('|', '\\|')
    authors = paper_info['authors'].replace('|', '\\|')
    year = str(paper_info['year'])
    publication = paper_info['publication'].replace('|', '\\|')
    findings = paper_info['findings'].replace('|', '\\|').replace('\n', ' ')

    return f"| {name} | {authors} | {year} | {publication} | {findings} |\n"


def save_markdown_table(table_content, output_dir):
    """Save the markdown table to a file"""
    output_path = output_dir / "papers_summary_table.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(table_content)
    return output_path


def clean_json_string(text):
    """Clean and extract JSON from the model's response"""
    # Find JSON content between triple backticks if present
    json_match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = text

    # Remove any non-JSON content before or after the main object
    json_str = re.search(r'(\{.*\})', json_str, re.DOTALL)
    if json_str:
        json_str = json_str.group(1)
    else:
        raise ValueError("No JSON object found in the response")

    return json_str.strip()


def parse_model_response(response_text):
    """Parse the model's response into structured data"""
    try:
        # Clean the JSON string
        json_str = clean_json_string(response_text)
        # Parse the cleaned JSON
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error parsing JSON: {str(e)}")
        st.code(response_text, language="json")
        # Return a default structure
        return {
            'title': 'Error parsing paper data',
            'authors': 'Unknown',
            'publication': 'N/A',
            'year': 'N/A',
            'findings': 'Error extracting data from paper',
            'methodology': '',
            'results': '',
            'limitations': ''
        }


def get_paper_summary_openai(text, api_key):
    """Get summary from OpenAI API with enhanced metadata extraction"""
    client = openai.OpenAI(api_key=api_key)

    system_prompt = """You are a research paper analyzer. Extract and summarize the following information from the paper in JSON format:
    {
        "title": "paper title",
        "authors": "author names (comma separated)",
        "publication": "publication name (journal/conference like IEEE)",
        "year": "publication year (just the number)",
        "findings": "Brief summary of key findings and contributions (2-3 sentences) along with the keywords",
        "methodology": "Brief description of methods used",
        "results": "Key results",
        "limitations": "Main limitations and future work"
    }

    Return ONLY the JSON object with these fields."""

    max_chunk_length = 12000
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please analyze this research paper text:\n\n{chunks[0]}"}
        ],
        temperature=0.5
    )

    structured_data = parse_model_response(response.choices[0].message.content)
    return structured_data, response.choices[0].message.content


def get_paper_summary_gemini(pdf_file, api_key):
    """Get summary directly from PDF using Gemini API with enhanced metadata extraction"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = """Analyze this research paper and provide the information in the following JSON format:
    {
        "title": "paper title",
        "authors": "author names (comma separated)",
        "publication": "publication name (journal/conference like IEEE)",
        "year": "publication year (just the number)",
        "findings": "Brief summary of key findings and contributions (2-3 sentences) along with the keywords",
        "methodology": "Brief description of methods used",
        "results": "Key results",
        "limitations": "Main limitations and future work"
    }

    Return ONLY the JSON object with these fields."""

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        uploaded_pdf = genai.upload_file(tmp_file_path)
        response = model.generate_content([prompt, uploaded_pdf])
        structured_data = parse_model_response(response.text)
        return structured_data, response.text
    finally:
        os.unlink(tmp_file_path)


def process_pdfs(pdf_files, model_choice, api_key, progress_bar, status_text):
    """Process multiple PDFs and save summaries incrementally"""
    output_dir = setup_output_directory()
    total_files = len(pdf_files)
    processed_files = []
    failed_files = []

    # Initialize markdown table
    table_content = create_markdown_table_header()

    for i, pdf_file in enumerate(pdf_files, 1):
        try:
            # Update progress
            progress_bar.progress(i / total_files)
            status_text.text(f"Processing {pdf_file.name} ({i}/{total_files})")

            # Process PDF
            if model_choice == "OpenAI GPT":
                text = extract_text_from_pdf(pdf_file)
                structured_data, full_summary = get_paper_summary_openai(text, api_key)
            else:
                pdf_file.seek(0)
                structured_data, full_summary = get_paper_summary_gemini(pdf_file, api_key)

            # Add to markdown table
            table_content += add_to_markdown_table(structured_data)

            # Save updated table
            table_path = save_markdown_table(table_content, output_dir)

            # Save full summary
            summary_filename = get_summary_filename(pdf_file.name)
            summary_path = save_summary(full_summary, summary_filename, output_dir)

            processed_files.append({
                'filename': pdf_file.name,
                'summary_path': summary_path,
                'table_path': table_path,
                'summary': full_summary,
                'structured_data': structured_data
            })

            # Small delay to prevent API rate limiting
            time.sleep(1)

        except Exception as e:
            failed_files.append({'filename': pdf_file.name, 'error': str(e)})
            continue

    return output_dir, processed_files, failed_files


# Streamlit UI
st.set_page_config(page_title="Batch PDF Summarizer", page_icon="📚", layout="wide")

st.title("📚 Batch PDF Summarizer")
st.write("Upload multiple research papers to get AI-generated summaries.")

# Sidebar for API settings
with st.sidebar:
    st.header("Settings")
    model_choice = st.radio("Choose AI Model:", ["Google Gemini", "OpenAI GPT"])

    if model_choice == "OpenAI GPT":
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        st.markdown("Get your OpenAI API key [here](https://platform.openai.com/api-keys)")
    else:
        api_key = st.text_input("Enter your Google AI API key:", type="password")
        st.markdown("Get your Google AI API key [here](https://makersuite.google.com/app/apikey)")
        st.info("💡 Gemini can process PDFs directly, resulting in better summaries!")

# File upload
pdf_files = st.file_uploader("Upload research papers (PDF)", type="pdf", accept_multiple_files=True)

if pdf_files:
    if not api_key:
        st.error(
            f"Please enter your {'OpenAI' if model_choice == 'OpenAI GPT' else 'Google AI'} API key in the sidebar.")
    else:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process button
        if st.button("Process PDFs"):
            output_dir, processed_files, failed_files = process_pdfs(
                pdf_files, model_choice, api_key, progress_bar, status_text
            )

            # Display results
            st.success(f"✅ Processing complete! Summaries saved to: {output_dir}")

            # Display markdown table
            if processed_files:
                st.subheader("📊 Summary Table")
                with st.expander("View Summary Table", expanded=True):
                    table_path = processed_files[0]['table_path']
                    with open(table_path, 'r') as f:
                        table_content = f.read()
                    st.markdown(table_content)
                    st.download_button(
                        label="Download Summary Table",
                        data=table_content.encode(),
                        file_name="papers_summary_table.md",
                        mime="text/markdown"
                    )

                # Display individual summaries
                st.subheader("📋 Detailed Summaries")
                for file in processed_files:
                    with st.expander(f"Summary for {file['filename']}"):
                        st.text(f"Saved to: {file['summary_path']}")
                        st.markdown(file['summary'])
                        st.download_button(
                            label=f"Download summary for {file['filename']}",
                            data=file['summary'].encode(),
                            file_name=file['summary_path'].name,
                            mime="text/plain"
                        )

            # Display any failures
            if failed_files:
                st.subheader("❌ Failed Files")
                for file in failed_files:
                    st.error(f"Failed to process {file['filename']}: {file['error']}")

# Add requirements info
st.markdown("---")
with st.expander("📦 Installation Requirements"):
    st.code("""
    pip install streamlit openai google-generativeai PyPDF2
    """)