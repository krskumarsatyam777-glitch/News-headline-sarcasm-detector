import spaces
import gradio as gr
from transformers import pipeline

# load fine-tuned DistilBERT model
classifier = pipeline(
    "text-classification",
    model="satyame639291/sarcasm-distilbert"
)

# sample headlines
examples = [
    "Shocking, Usain bolt realisies he can use legs for walking",
    "Increasing temperature is a sign for global warming",
    "ford develops new suv that runs purely on gasoline",
    "how to live to be 110"
]
@spaces.GPU
def predict(headline):
    if not headline.strip():
        return "⚠️ Please enter a news headline"

    result = classifier(headline)[0]
    label = result["label"]
    confidence = result["score"]

    if label == "Sarcastic":
        return (
            f"🎭 **Satirical News Headline**\n\n"
            f"**Confidence:** {confidence:.2%}\n\n"
            f"The model predicts that this headline resembles **sarcastic news**, "
        )
    else:
        return (
            f"📰 **Genuine News Headline**\n\n"
            f"**Confidence:** {confidence:.2%}\n\n"
            f"The model predicts that this headline resembles **genuine news reporting** "
        )

with gr.Blocks(title="News Headline Sarcasm Detection") as demo:
    gr.Markdown("# 🗞️ News Headline Sarcasm Detection")
    gr.Markdown(
        "This app uses a fine-tuned DistilBERT model to determine whether a "
        "news headline is satirical or a genuine news report."
    )

    headline_input = gr.Textbox(
        label="Enter a News Headline",
        placeholder="Type the news headline here"
    )

    gr.Examples(
        examples=examples,
        inputs=headline_input,
        label="Try a Sample Headline"
    )

    analyze_btn = gr.Button("🔍 Analyze Headline", variant="primary")
    output = gr.Markdown()

    analyze_btn.click(fn=predict, inputs=headline_input, outputs=output)

    with gr.Accordion("ℹ️ About this App", open=False):
        gr.Markdown(
            """
            This app detects whether a news headline is likely **satirical** (like The Onion)
            or **genuine** (like real news reporting), using a fine-tuned **DistilBERT** model.

            **How it works:** Enter any headline, or try one of the sample headlines above,
            and the model will classify it along with a confidence score.

            **Note:** The model was trained on a specific dataset of headlines and may not
            generalize perfectly to all writing styles or topics outside its training data.

            [View the full project, dataset details, and model comparison on GitHub →](https://github.com/krskumarsatyam777-glitch/News-headline-sarcasm-detector)
            """
        )

demo.launch()