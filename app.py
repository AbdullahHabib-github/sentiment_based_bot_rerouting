from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
from utilities import select_emotion
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("API Key not found. Please check your .env file.")

# Set the API key for the OpenAI client
client = OpenAI(api_key=openai_api_key)

# Store LLM responses and current emotion in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "Neutral"  # Default emotion

# Sidebar for displaying the current emotion
st.sidebar.header("Current Emotion")
emotion_display = st.sidebar.empty()  # Placeholder for emotion text
emotion_display.write(st.session_state.current_emotion)

# # Display chat messages
for message in st.session_state.messages:
    avatar = None
    if message["role"] == "assistant":
        avatar = '🤖'
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

# Handle user input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get the current emotion
    meta_emotion = select_emotion(prompt)
    st.session_state.current_emotion = meta_emotion["Emotion"]  # Update the emotion state
    emotion_display.write(st.session_state.current_emotion)
    assistant_id = meta_emotion["Assistant ID"]
    # Generate a response from the assistant
    with st.chat_message("assistant", avatar='🤖'):
        loading_text = st.empty()  # Placeholder for dynamic updates
        loading_text.text("Thinking...")  # Show initial loading message

        # Call the API to get a response
        run = client.beta.threads.create_and_run(
            assistant_id=assistant_id,
            thread={
                "messages": st.session_state.messages
            }
        )

        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=run.thread_id,
                run_id=run.id
            )
            if run.status in ['cancelled', 'failed', 'completed', 'expired']:
                break

        loading_text.empty()  # Remove the "Thinking..." message

        # Handle errors or fetch the generated response
        if run.last_error:
            response = run.last_error
        else:
            thread_messages = client.beta.threads.messages.list(thread_id=run.thread_id)
            response = thread_messages.data[0].content[0].text.value

        # Display the response directly using st.write()
        # st.write(response)

        # Append the assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

        st.rerun()
