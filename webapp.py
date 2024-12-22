from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr

# Load the saved model and tokenizer
model = load_model("reviews.h5")
tokenizer = joblib.load("tokenizer.pkl")

# Define the predictive system with an updated progress bar
def predictive_system(Review):
    sequences = tokenizer.texts_to_sequences([Review])
    paddedSequence = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(paddedSequence)
    
    # Calculate the probability of positive sentiment
    positive_prob = prediction[0][0]
    
    if positive_prob > 0.5:
        percentage = int(positive_prob * 100)
        progress_label = f"{percentage}% Positive"
        bar_color = "green"
    else:
        percentage = int((1 - positive_prob) * 100)  # Show percentage of negative sentiment
        progress_label = f"{percentage}% Negative"
        bar_color = "red"
    
    # Generate the progress bar HTML with color based on sentiment
    progress_bar_html = f"""
    <div style="width: 100%; background-color: lightgray; border-radius: 5px;">
        <div style="width: {percentage}%; background: linear-gradient(to right, red, green); padding: 10px; color: white; text-align: center; border-radius: 5px;">
            {progress_label}
        </div>
    </div>
    """
    
    # Return the sentiment label and the progress bar
    sentiment_text = "Review is Positive" if positive_prob > 0.5 else "Review is Negative"
    
    return sentiment_text, progress_bar_html

# Set up the Gradio app
title = "Amazon Product Review Sentiment Analysis Application"

# Use 'inputs' and 'outputs' for the main prediction
with gr.Blocks(title="Review Analysis") as app:  # Set the tab title here
    # Center the heading using HTML
    gr.HTML(f"<h1 style='text-align: center;'>{title}</h1>")
    
    # Input text for the review
    review_input = gr.Textbox(label="Enter your review here")
    
    # Button to submit the review for prediction
    submit_button = gr.Button("Submit")
    
    # Output area for the prediction result and the progress bar
    output_text = gr.Textbox(label="Sentiment")
    output_progress_bar = gr.HTML(label="Sentiment Probability")

    # Submit button functionality (for prediction and progress bar)
    submit_button.click(predictive_system, inputs=review_input, outputs=[output_text, output_progress_bar])

# Launch the app
app.launch(share=True)
