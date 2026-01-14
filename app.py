import streamlit as st
import joblib

label_map = {
    0: "Sadness 😢",
    1: "Anger 😠",
    2: "Love ❤️",
    3: "Surprise 😮",
    4: "Fear 😨",
    5: "Joy 😊"
}


# ---------- Page Config ----------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊",
    layout="centered"
)

st.markdown("""
<style>
    /* Hide default sidebar arrow */
    [data-testid="collapsedControl"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Load Model & Vectorizer ----------

@st.cache_resource
def load_model():
    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()


# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .title {
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        color: #b0b3b8;
        text-align: center;
        margin-bottom: 30px;
    }
    .emotion-box {
        background-color: #1f2933;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        color: #00ffcc;
    }
</style>
""", unsafe_allow_html=True)

# ---------- UI ----------
st.markdown(
    "<p style='text-align:center;color:#b0b3b8;'>"
    "by Nihal Ahemad Khan"
    "</p>",
    unsafe_allow_html=True
)



st.markdown("<div class='title'>Emotion Detection App</div>", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## 👨‍💻 About the App")

    st.write("**Developer:** Nihal Ahemad Khan")
    st.write("**Project:** Emotion Detection using NLP")
    st.write("**Model:** Logistic Regression")
    st.write("**Text Representation:** TF-IDF Vectorizer")
    st.write("**Emotions Supported:** 6 classes")

    st.markdown("---")

    st.markdown("## ℹ️ How it works")
    st.write(
        "Enter a sentence describing your feelings. "
        "The model analyzes the text and predicts the most likely emotion."
    )

    st.markdown("---")

    st.markdown("## 📬 Need more info?")
    st.write(
        "If you need more details about the model, dataset, "
        "or want to collaborate, feel free to reach out."
    )
    st.markdown(
    """
    📧 **Email:**  
    <a href="mailto:nihalahemad2003@gmail.com">
        nihalahemad2003@gmail.com
    </a>

    💼 **LinkedIn:**  
    <a href="https://www.linkedin.com/in/nihal-ahemad-khan" target="_blank">
        linkedin.com/in/nihal-ahemad-khan
    </a>
    """,
    unsafe_allow_html=True
)




st.markdown("<div class='subtitle'>Logistic Regression + NLP</div>", unsafe_allow_html=True)

st.write("")
text_input = st.text_area(
    "💬 Enter your text",
    height=120,
    placeholder="Type how you feel..."
)

# ---------- Prediction ----------
if st.button("🔍 Analyze Emotion"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([text_input])
        pred_label = model.predict(X)[0]
        prediction = label_map[pred_label]

        st.write("")
        st.markdown(
            f"<div class='emotion-box'>Predicted Emotion: {prediction}</div>",
             unsafe_allow_html=True

        )

# ---------- Footer ----------
st.write("")
st.markdown(
    "<hr style='margin-top:40px;'>"
    "<p style='text-align:center;color:gray;'>"
    "Built by <b>Nihal Ahemad Khan</b> | NLP Emotion Detection"
    "</p>",
    unsafe_allow_html=True
)

