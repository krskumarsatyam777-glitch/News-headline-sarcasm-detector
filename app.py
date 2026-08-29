import streamlit as st
import os
from transformers import pipeline

#page configuration

st.set_page_config(
    page_title="News Headline Sarcasm Detection",
    page_icon="🗞️",
    layout="centered"
)

# load fine-tuned DistilBRET model

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="satyame639291/sarcasm-distilbert",
        token=os.environ.get("HF_TOKEN")
    )

classifier = load_model()

# App title

st.title("News Headline Sarcasm Detection")
st.markdown(""" This app uses a fine-tuned DistilBRET model to determine
whether a news headline is satirical or genuine news report.""")

st.divider()

st.subheader("Try a Sample Headline")

examples = [
    "Shocking, Usain bolt realisies he can use legs for walking",
    "Increasing temperature is a sign for global warming",
    "ford develops new suv that runs purely on gasoline",
    "how to live to be 110"
]

cols = st.columns(2)
for i,example in enumerate(examples):
    if cols[i%2].button(example,use_container_width=True):
        st.session_state["headline"] = example

        st.divider()

#user unput

headline = st.text_input(
    "Enter a News Headline",
value=st.session_state.get("headline",""),
placeholder="Type the news headline here"
)

#prediction

if st.button("🔍 Analyze Headline", type="primary", use_container_width =True):

    if not headline.strip():
        st.warning("Please enter a news headline")

    else:
        with st.spinner("Analyzing..."):
            result = classifier(headline)[0]
            label = result["label"]
            confidence = result["score"]

            st.divider()
            st.subheader("Prediction")

            if label == "Sarcastic":
                st.error("🎭 **Satirical News Headline**")
                st.write(f"**Confidence:** {confidence:.2%}")
                st.progress(confidence)

                st.markdown(""" the model predicts that this 
                headline resembles **sarcastic news**, similar to 
                articles published by **The onion**""")

            else:
                st.success("📰 **Genuine News Headline**")
                st.write(f"**Confidence:** {confidence:.2%}")
                st.progress(confidence)

                st.markdown("""
                The model predicts that this headline resembles **genuine news reporting**,
                similar to articles published by **HuffPost**.
                """)

st.divider()

#about model

with st.expander("ℹ️ About this App"):

    st.markdown(
        """
        This app detects whether a news headline is likely **satirical** (like The Onion) 
        or **genuine** (like real news reporting), using a fine-tuned **DistilBERT** model.

        **How it works:** Enter any headline, or try one of the sample headlines above, 
        and the model will classify it along with a confidence score.

        **Note:** The model was trained on a specific dataset of headlines and may not 
        generalize perfectly to all writing styles or topics outside its training data.

        [View the full project, dataset details, and model comparison on GitHub →](https://github.com/krskumarsatyam777-glitch/News-headline-sarcasm-detector)
        """)