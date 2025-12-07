import streamlit as st
st.set_page_config(page_title="SMS Spam Detection", page_icon="📩")

import pickle
import re
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.exceptions import NotFittedError

# --- NLTK setup (sirf stopwords chahiye) ---
nltk.download("stopwords")

ps = PorterStemmer()

def transform_text(text: str) -> str:
    # lowercase
    text = text.lower()

    # simple regex tokenization (no punkt needed)
    tokens = re.findall(r"\b\w+\b", text)

    # remove non-alphanumeric (waise regex se already clean hai)
    tokens = [tok for tok in tokens if tok.isalnum()]

    # remove stopwords
    sw = set(stopwords.words("english"))
    tokens = [tok for tok in tokens if tok not in sw]

    # stemming
    stemmed = [ps.stem(tok) for tok in tokens]

    return " ".join(stemmed)

@st.cache_resource
def load_model_and_vectorizer():
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_model_and_vectorizer()

# --- UI ---
st.title("📩 SMS Spam Detection App")
st.write("Check whether a message is **Spam** or **Not Spam (Ham)** using your ML model.")

input_sms = st.text_area("✍️ Enter SMS text here", height=150)

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message!")
    else:
        try:
            # 1. preprocess
            transformed_sms = transform_text(input_sms)

            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 3. predict
            result = model.predict(vector_input)[0]

            # 4. show result
            if result == 1:
                st.error("⚠️ Spam Message Detected")
            else:
                st.success("✅ Not Spam (Ham) Message")

        except NotFittedError:
            st.error(
                "❌ The loaded model is not fitted.\n"
                "Train your model (clf) and save it with:\n"
                "pickle.dump(clf, open('model.pkl', 'wb'))"
            )
