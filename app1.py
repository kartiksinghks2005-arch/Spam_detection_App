from groq import Groq
import streamlit as st
import joblib
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import PyPDF2

# ---------- GROQ CLIENT ----------
client = Groq(api_key="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")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Message Safety Analyzer",
    page_icon="🛡️",
    layout="wide"
)

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------- STYLE ----------
st.markdown("""
<style>
.stButton>button {
background-color:#ff4b4b;
color:white;
border-radius:10px;
}
</style>
""", unsafe_allow_html=True)


# ---------- HELPER FUNCTIONS ----------
def show_confidence(prob):

    st.progress(prob)

    if prob > 0.8:
        st.error("🔴 High Risk")
    elif prob > 0.5:
        st.warning("🟠 Medium Risk")
    else:
        st.success("🟢 Low Risk")


def highlight_words(text, words):

    for word in words:
        text = text.replace(
            word,
            f"<span style='color:red;font-weight:bold'>{word}</span>"
        )

    return text


# ---------- LLM ANALYSIS ----------
def llm_analysis(message):

    prompt = f"""
You are a cybersecurity AI assistant.

Analyze the message and return the result in EXACTLY this format:

Spam/Safe | Risk Level | Short Reason

Example:
Spam | High | Contains prize scam language

Message:
{message}
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.choices[0].message.content.strip()

        return result

    except Exception as e:

        return f"LLM Error: {e}"

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛡️ AI Analyzer")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📩 Single Message", "📂 Bulk Scanner", "📊 Dashboard", "ℹ️ About"]
)

# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title("🛡️ AI Message Safety Analyzer")

    st.subheader("Hybrid ML + LLM Spam Detection System")

    col1, col2 = st.columns(2)

    col1.info("✔ Detect Spam Messages")
    col1.info("✔ Show Spam Probability")
    col1.info("✔ WordCloud Visualization")
    col1.info("✔ Multi File Scanner")

    col2.success("🚀 Built with")
    col2.write("- Machine Learning")
    col2.write("- NLP")
    col2.write("- Streamlit")
    col2.write("- Groq LLM")


# =========================================================
# SINGLE MESSAGE
# =========================================================

elif page == "📩 Single Message":

    st.header("📩 Check Single Message")

    message = st.text_area("Enter message")

    if st.button("Analyze Message"):

        if message.strip() == "":

            st.warning("Please enter a message")

        else:

            data = vectorizer.transform([message]).toarray()

            prob = model.predict_proba(data)[0][1]

            prediction = model.predict(data)[0]

            st.session_state["last_message"] = message

            st.subheader("Result")

            if prediction == "spam":

                st.error("🚨 Spam Message Detected")

            else:

                st.success("✅ Safe Message")

            st.subheader("📊 Confidence")

            show_confidence(prob)

            words = message.lower().split()

            vocab = vectorizer.get_feature_names_out()

            suspicious = [w for w in words if w in vocab][:8]

            st.subheader("⚠️ Suspicious Words")

            highlighted = highlight_words(message, suspicious)

            st.markdown(highlighted, unsafe_allow_html=True)

            # WORDCLOUD
            st.subheader("☁️ WordCloud")

            wc = WordCloud(
                width=600,
                height=300,
                background_color="black"
            ).generate(message)

            fig, ax = plt.subplots(figsize=(6,3))

            ax.imshow(wc)

            ax.axis("off")

            st.pyplot(fig)

    if "last_message" in st.session_state:

        st.subheader("🧠 AI Explanation")

        if st.button("Explain with AI"):

            with st.spinner("AI analyzing message..."):

                result = llm_analysis(st.session_state["last_message"])

            st.info(result)


# =========================================================
# BULK SCANNER
# =========================================================

elif page == "📂 Bulk Scanner":

    st.header("📂 Upload file for bulk spam detection")

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "txt", "json", "pdf"]
    )

    if uploaded_file:

        file_type = uploaded_file.name.split(".")[-1]

        # ---------- READ FILE ----------

        if file_type == "csv":

            df = pd.read_csv(uploaded_file)

        elif file_type == "xlsx":

            df = pd.read_excel(uploaded_file)

        elif file_type == "txt":

            text = uploaded_file.read().decode("utf-8")

            messages = text.split("\n")

            df = pd.DataFrame(messages, columns=["message"])

        elif file_type == "json":

            data = json.load(uploaded_file)

            df = pd.DataFrame(data)

        elif file_type == "pdf":

            reader = PyPDF2.PdfReader(uploaded_file)

            text = ""

            for page in reader.pages:

                text += page.extract_text()

            messages = text.split("\n")

            df = pd.DataFrame(messages, columns=["message"])

        else:

            st.error("Unsupported file format")

            st.stop()

        if "message" not in df.columns:

            st.error("File must contain column 'message'")

            st.stop()

        # ---------- SPAM DETECTION ----------

        vectors = vectorizer.transform(df["message"]).toarray()

        df["Prediction"] = model.predict(vectors)

        df["Spam Probability"] = model.predict_proba(vectors)[:,1]

        # ---------- AI EXPLANATION ----------

        with st.spinner("Running AI analysis..."):

            df["AI Explanation"] = df["message"].apply(llm_analysis)

        st.success("Scan Completed")

        st.dataframe(df.head())

        # ---------- SUMMARY ----------

        st.header("📊 Summary")

        summary = df["Prediction"].value_counts()

        spam = summary.get("spam",0)

        ham = summary.get("ham",0)

        fig, ax = plt.subplots()

        ax.pie(
            [ham, spam],
            labels=["Safe","Spam"],
            autopct="%1.1f%%",
            colors=["#2ecc71","#e74c3c"]
        )

        st.pyplot(fig)

        # ---------- TREND CHART ----------

        st.subheader("Spam Trend")

        st.bar_chart(df["Prediction"].value_counts())

        # ---------- DOWNLOAD ----------

        csv = df.to_csv(index=False).encode()

        st.download_button(
            "⬇ Download Results",
            csv,
            "results.csv"
        )


# =========================================================
# DASHBOARD
# =========================================================

elif page == "📊 Dashboard":

    st.header("📊 Model Insights")

    st.metric("Model Type","Naive Bayes")

    st.metric("Vectorizer","CountVectorizer")

    st.success("ML pipeline running correctly")


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.header("About Project")

    st.write("""
AI Message Safety Analyzer

Features:

• ML Spam Detection  
• AI Explanation  
• Suspicious Word Highlighting  
• WordCloud Visualization  
• Multi File Bulk Scanner  
• Spam Trend Analytics
""")

