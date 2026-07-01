import streamlit as st
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
        model="satyame639291/sarcasm-distilbert"
    )

classifier = load_model()

# App title

st.title("News Headline Sarcasm Detection")
st.markdown(""" This app uses a fine-tuned DistilBRET model to determine
whether a news headline is satirical or genuine news report.""")

st.divider()

st.subheader("Try a Sample Headline")

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

with st.expander("ℹ️ About this Model"):

    st.markdown(
        """
        ### Model
    
        - **Architecture:** Fine-tuned DistilBERT (`distilbert-base-uncased`)
        - **Task:** News Headline Classification
    
        ### Dataset
    
        **News Headlines Dataset for Sarcasm Detection**
    
        - **Total Headlines:** 28,503
        - **Satirical News:** The Onion
        - **Genuine News:** HuffPost
    
        ### Performance
    
        | Model | Accuracy | F1-Score |
        |:------|:---------:|:--------:|
        | TF-IDF + Logistic Regression | **77.34%** | **0.7729** |
        | Fine-tuned DistilBERT | **92.10%** | **0.9209** |
        """)

    st.markdown("---")
    
    
