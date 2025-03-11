import gradio as gr

with gr.Blocks(fill_height=True) as demo:
    with gr.Sidebar():
        gr.Markdown("# Inference Provider")
        gr.Markdown(
            "This Space showcases the meta-llama/Llama-3.2-1B model, served by the hf-inference API. "
            "Sign in with your Hugging Face account to use this API."
        )
        # LoginButton will prompt the user to sign in with their Hugging Face account.
        button = gr.LoginButton("Sign in")
    
    # Load the model from Hugging Face using the hf-inference provider.
    # The accept_token parameter ensures that the signed-in user's token is used.
    gr.load("models/meta-llama/Llama-3.2-1B", accept_token=button, provider="hf-inference")
    
demo.launch()
