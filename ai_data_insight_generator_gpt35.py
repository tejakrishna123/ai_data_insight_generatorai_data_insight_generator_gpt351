
import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Data Insight Generator", layout="wide")

st.title("📊 AI-Powered Data Insight Generator")
st.markdown("Upload your dataset and let AI help you discover insights.")

# Input OpenAI API key securely
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your dataset", df.head())

    st.write("### Column Summary")
    st.write(df.describe(include='all'))

    if openai_api_key:
        # Set OpenAI key
        openai.api_key = openai_api_key

        with st.spinner("Analyzing your dataset with AI..."):
            prompt = f"You are a data analyst. Based on the dataset: {df.head(100).to_csv(index=False)}, summarize key insights, trends, and potential business implications."

            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a skilled data analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=500
                )
                ai_insight = response.choices[0].message.content
                st.markdown("### 📈 AI-Generated Insights")
                st.success(ai_insight)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.write("### 📊 Quick Data Visualizations")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_columns) >= 2:
        col1 = st.selectbox("X-axis", numeric_columns)
        col2 = st.selectbox("Y-axis", numeric_columns, index=1)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Upload a dataset with at least 2 numeric columns to see visualizations.")
