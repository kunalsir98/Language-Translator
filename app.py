import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌍",
    layout="centered"
)

# ----------------- CUSTOM CSS -----------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }

    .main {
        background: transparent;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
    }

    h1 {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #d1d5db;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }

    textarea {
        border-radius: 12px !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 0.6rem;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        transition: 0.2s ease-in-out;
    }

    .output-box {
        background: rgba(0, 0, 0, 0.4);
        padding: 20px;
        border-radius: 14px;
        color: #e5e7eb;
        font-size: 1.1rem;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- ENV & MODEL -----------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following message into {language}:"),
        ("user", "{text}")
    ]
)

chain = prompt | model | parser

# ----------------- UI -----------------
st.markdown("<h1>🌍 AI Language Translator</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Powered by Groq • LLaMA 3.3 • LangChain</p>",
    unsafe_allow_html=True
)

with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    text = st.text_area(
        "✍️ Enter text",
        placeholder="Hello, my name is Kunal and I am a Gen AI Developer...",
        height=120
    )

    language = st.selectbox(
        "🌐 Translate to",
        ["Hindi", "French", "Spanish", "German", "Japanese", "Chinese", "Korean"]
    )

    translate_btn = st.button("🚀 Translate")

    if translate_btn:
        if text.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            with st.spinner("✨ Translating with AI..."):
                result = chain.invoke(
                    {"language": language, "text": text}
                )

            st.markdown("### ✅ Translated Output")
            st.markdown(
                f"<div class='output-box'>{result}</div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- FOOTER -----------------
st.markdown(
    "<p style='text-align:center;color:#9ca3af;margin-top:30px;'>"
    "Built with ❤️ by Kunal | Gen AI Project</p>",
    unsafe_allow_html=True
)
