import os
import streamlit as st
import pandas as pd
from data_agent import DataAnalystAgent

st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")
st.title("📊 Data Analyst Agent")
st.caption("Upload a CSV, ask a question in plain English, and the agent writes and runs the pandas code for you — powered by a local LLM (Llama 3.1 8B via Ollama).")

with st.sidebar:
    st.header("Setup")
    model = st.text_input("Ollama model", value="llama3.1:8b")
    uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
    use_sample = st.checkbox("Use sample sales dataset instead", value=uploaded_file is None)

csv_path = None
if uploaded_file is not None and not use_sample:
    with open("uploaded.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    csv_path = "uploaded.csv"
elif use_sample:
    csv_path = "sales_data.csv"

if csv_path and os.path.exists(csv_path):
    if "agent" not in st.session_state or st.session_state.get("csv_path") != csv_path:
        st.session_state.agent = DataAnalystAgent(csv_path, model=model)
        st.session_state.csv_path = csv_path

    agent = st.session_state.agent

    with st.expander("Preview data"):
        st.dataframe(agent.df.head(20))
        st.text(f"{agent.df.shape[0]} rows × {agent.df.shape[1]} columns")

    question = st.text_input("Ask a question about your data", placeholder="e.g. Which region has the highest average profit margin?")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            response = agent.ask(question)

        st.subheader("Answer")
        if response["chart_path"] and os.path.exists(response["chart_path"]):
            st.image(response["chart_path"])
        st.write(response["result"])

        with st.expander("Generated code"):
            st.code(response["code"], language="python")
        st.caption(f"Attempts: {response['attempts']}")
else:
    st.info("Upload a CSV or check 'Use sample sales dataset' to get started.")
