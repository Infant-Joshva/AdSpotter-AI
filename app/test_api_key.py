# Chat Bot
# import streamlit as st
# import google.generativeai as genai
# # genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# # st.write("Key Loaded:", "GEMINI_API_KEY" in st.secrets)

import google.generativeai as genai
import streamlit as st

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

models = genai.list_models()
for m in models:
    print(m.name)

