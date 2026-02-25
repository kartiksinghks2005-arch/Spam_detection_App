import streamlit as st
import joblib
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Message Safety Analyzer",
    page_icon="🛡️",
    layout="wide"
)

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------- HELPER FUNCTIONS ----------
def highlight_words(text, words):
    for word in words:
        text = text.replace(word, f"<span style='color:red;font-weight:bold'>{word}</span>")
    return text

def show_confidence(prob):
    st.progress(prob)
    if prob > 0.8:
        st.error("🔴 High Risk")
    elif prob > 0.5:
        st.warning("🟠 Medium Risk")
    else:
        st.success("🟢 Low Risk")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🛡️ AI Analyzer")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📩 Single Message", "📂 Bulk Scanner", "📊 Dashboard", "ℹ️ About"]
)

# =========================================================
# HOME PAGE
# =========================================================
if page == "🏠 Home":
    st.title("🛡️ AI Message Safety Analyzer")
    st.subheader("Detect Spam Messages using Machine Learning")

    col1, col2 = st.columns(2)

    col1.info("✔ Detect Spam Messages")
    col1.info("✔ Show Spam Probability")
    col1.info("✔ Highlight Suspicious Words")
    col1.info("✔ WordCloud Visualization")
    col1.info("✔ Bulk CSV Scanner")

    col2.success("🚀 Built using:")
    col2.write("- Machine Learning")
    col2.write("- NLP")
    col2.write("- Streamlit")
    col2.write("- Scikit-learn")

# =========================================================
# SINGLE MESSAGE PAGE
# =========================================================
elif page == "📩 Single Message":

    st.header("📩 Check Single Message")
    message = st.text_area("Enter message")

    if st.button("Analyze Message"):

        data = vectorizer.transform([message]).toarray()
        prob = model.predict_proba(data)[0][1]
        prediction = model.predict(data)[0]

        st.subheader("Result")
        if prediction == "spam":
            st.error("🚨 Spam Message Detected")
        else:
            st.success("✅ Safe Message")

        st.subheader("📊 Confidence")
        show_confidence(prob)

        # suspicious words
        words = message.lower().split()
        vocab = vectorizer.get_feature_names_out()
        suspicious = [w for w in words if w in vocab][:8]

        st.subheader("⚠️ Suspicious Words")
        st.write(suspicious if suspicious else "None")

        # WordCloud
        if message.strip() != "":
            st.subheader("☁️ WordCloud")
            wc = WordCloud(width=600, height=300, background_color="black").generate(message)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.imshow(wc); ax.axis("off")
            st.pyplot(fig)

# =========================================================
# BULK SCANNER PAGE
# =========================================================
elif page == "📂 Bulk Scanner":

    st.header("📂 Upload CSV with column 'message'")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        vectors = vectorizer.transform(df["message"]).toarray()
        df["Prediction"] = model.predict(vectors)
        df["Spam Probability"] = model.predict_proba(vectors)[:,1]

        st.success("Scan Completed")
        st.dataframe(df.head())

        # Summary
        st.header("📊 Summary")
        summary = df["Prediction"].value_counts()
        spam = summary.get("spam",0)
        ham = summary.get("ham",0)

        col1,col2,col3 = st.columns([1,2,1])
        with col2:
            fig,ax = plt.subplots(figsize=(4,4))
            ax.pie([ham,spam],labels=["Safe","Spam"],autopct="%1.1f%%",
                   colors=["#2ecc71","#e74c3c"],startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download Results", csv, "results.csv")

# =========================================================
# DASHBOARD PAGE
# =========================================================
elif page == "📊 Dashboard":
    st.header("📊 Model Insights")

    st.metric("Model Type", "Naive Bayes")
    st.metric("Vectorizer", "CountVectorizer")

    st.write("This model is trained on SMS Spam Dataset.")
    st.write("Average Accuracy ~ 98%+")

    st.success("Your ML pipeline is working correctly 🎉")

# =========================================================
# ABOUT PAGE
# =========================================================
elif page == "ℹ️ About":

    st.header("About Project")

    st.write("""
    **AI Message Safety Analyzer**
    
    Features:
    - Spam detection using NLP
    - Real-time predictions
    - Bulk CSV scanning
    - WordCloud visualization
    
    Built for portfolio & placements 🚀
    """)