import gradio as gr
import boto3
import json

# Set up the SageMaker runtime client with your region.
runtime = boto3.client('sagemaker-runtime', region_name='eu-north-1')

# Your SageMaker endpoint name.
endpoint_name = "jumpstart-dft-meta-textgeneration-l-20250311-220050"

def chat_function(user_message, history):
    """
    Sends a message to the SageMaker endpoint and returns the updated chat history.
    """
    # You can modify this to combine conversation history if desired.
    payload = {"inputs": user_message}
    
    # Invoke the endpoint.
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    
    # Process the response.
    result = json.loads(response["Body"].read().decode())
    
    # Adjust depending on your endpoint's response format.
    # For example, if the response is a dict with a "generated_text" key:
    bot_reply = result.get("generated_text", "") if isinstance(result, dict) else str(result)
    
    # Append the latest interaction to history.
    history.append((user_message, bot_reply))
    return history, history

# Build a Gradio Blocks interface with a chat component.
with gr.Blocks() as demo:
    gr.Markdown("## Chat with meta-llama/Llama-3.2-1B")
    chatbot = gr.Chatbot()
    # State holds the conversation history.
    state = gr.State([])

    txt = gr.Textbox(show_label=False, placeholder="Type a message and hit enter")
    txt.submit(chat_function, inputs=[txt, state], outputs=[chatbot, state])

# Launch the Gradio app.
demo.launch()
