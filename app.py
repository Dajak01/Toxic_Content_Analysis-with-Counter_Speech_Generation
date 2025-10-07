
import streamlit as st
from utils.model_utils import analyze_toxicity_with_bert, generate_counter_response, analyze_with_openai

def main():
    st.set_page_config(page_title="Toxicity and Counter Speech App", layout="wide")
    st.image("logo.png", width=100)

    st.title("Toxicity Analysis and Counter Speech Generation")
    col1, col2 = st.columns(2)

    # Initialize session state variables if they don't exist
    if 'openai_level' not in st.session_state:
        st.session_state['openai_level'] = ""
    if 'openai_response' not in st.session_state:
        st.session_state['openai_response'] = ""
    if 'custom_toxicity' not in st.session_state:
        st.session_state['custom_toxicity'] = ""
    if 'custom_response' not in st.session_state:
        st.session_state['custom_response'] = ""

    with col1:
        st.header("OpenAI Analysis")
        openai_input = st.text_area("Enter text to analyze for toxicity with OpenAI:", "Type your text here...", key='openai_input')
        if st.button("Analyze with OpenAI", key='analyze_openai'):
            level, response = analyze_with_openai(openai_input, "sk-proj-5V1UsF8sgJqRmQ87CgtQT3BlbkFJV8WjkYtSa8PZfpoOUil6")
            st.session_state['openai_level'] = level
            st.session_state['openai_response'] = response
        
        # Display results
        if st.session_state['openai_level']:
            st.write(f"Toxicity Level: {st.session_state['openai_level']}")
            if 'non-toxic' not in st.session_state['openai_level'].lower():
                st.write(f"Counter Response: {st.session_state['openai_response']}")

    with col2:
        st.header("Custom Model Analysis")
        custom_model_input = st.text_area("Enter text to analyze for toxicity with Custom Model:", "Type your text here...", key='custom_model_input')
        if st.button("Analyze with Custom Model", key='analyze_custom'):
            toxicity = analyze_toxicity_with_bert(custom_model_input)
            response = generate_counter_response(custom_model_input, toxicity)
            st.session_state['custom_toxicity'] = toxicity
            st.session_state['custom_response'] = response
        
        # Display results
        if st.session_state['custom_toxicity']:
            st.write(f"Toxicity Level: {st.session_state['custom_toxicity']}")
            if 'non-toxic' not in st.session_state['custom_toxicity'].lower():
                st.write(f"Counter Response: {st.session_state['custom_response']}")

if __name__ == '__main__':
    main()

