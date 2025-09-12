from src.qwen2_v2 import my_qwen2_cpu as my_qwen2
import gradio as gr

# Get list of supported models from the Qwen2 implementation
dropdown_options = my_qwen2.SUPPORT_MODELS


def generate_response(model_name, message, chat_history):
    """
    Generate a response using the selected Qwen2 model and update chat history

    Args:
        model_name (str): Name of the selected model
        message (str): User input message
        chat_history (list): List of tuples containing previous (user_msg, bot_response)

    Returns:
        tuple: Empty string (to clear input) and updated chat history
    """
    # Ignore empty messages
    if not message.strip():
        return "", chat_history

    # Construct full model name with prefix
    full_model_name = f"Qwen/{model_name}"

    try:
        # Initialize the Qwen2 model with specified parameters
        model = my_qwen2.Qwen2(full_model_name, max_new_tokens=100)

        # Generate response from the model based on user input
        response = model.generate(message)

        # Add current conversation turn to chat history
        chat_history.append((message, response))

        # Return empty string to clear input box and updated chat history
        return "", chat_history
    except Exception as e:
        # Handle and display any errors that occur during generation
        error_msg = f"Error: {str(e)}"
        chat_history.append((message, error_msg))
        return "", chat_history


def clear_chat():
    """
    Clear the chat history by returning an empty list

    Returns:
        list: Empty list to reset chat history
    """
    return []


if __name__ == "__main__":
    # Create a Gradio Blocks interface with flexible layout
    with gr.Blocks(title="Qwen2 chat demo", theme=gr.themes.Monochrome()) as UI:
        # Add title and description using Markdown with limited margin
        gr.Markdown("# Qwen2 chat demo", elem_classes="header")
        gr.Markdown("This is a chat bot based on handwritten Qwen", elem_classes="subheader")

        # Dropdown component for model selection
        model_selector = gr.Dropdown(
            choices=dropdown_options,
            label="Select model",
            value=dropdown_options[0] if dropdown_options else None,
            elem_classes="model-selector",
        )

        # Chatbot component with scrollable area
        chatbot = gr.Chatbot(
            label="Chat history",
            height=300,  # Reduced height to ensure controls stay visible
            container=True,
            elem_classes="chat-history",
        )

        # Text input component for user messages
        msg = gr.Textbox(
            lines=2,  # Reduced lines to save space
            placeholder="Type your message...",
            label="Input",
            elem_id="chat-input",
        )

        # Create a row for action buttons
        with gr.Row(elem_classes="button-row"):
            clear_btn = gr.Button("Clear chat")
            submit_btn = gr.Button("Send", variant="primary")

        # Configure button click event
        submit_btn.click(generate_response, inputs=[model_selector, msg, chatbot], outputs=[msg, chatbot])

        # Configure clear button
        clear_btn.click(clear_chat, inputs=[], outputs=[chatbot])

        # Custom CSS to ensure all elements are visible
        UI.css = """
        /* Ensure the entire interface fits in viewport */
        .gradio-container {
            min-height: auto !important;
            max-height: 100vh !important;
            padding: 1rem !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }
        
        /* Header styling with reduced margins */
        .header {
            margin: 0 0 0.5rem 0 !important;
            padding: 0 !important;
        }
        
        .subheader {
            margin: 0 0 1rem 0 !important;
            padding: 0 !important;
            font-size: 0.9rem !important;
        }
        
        /* Model selector with reduced margin */
        .model-selector {
            margin-bottom: 0.8rem !important;
        }
        
        /* Scrollable chat history */
        .chat-history {
            overflow-y: auto !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
            margin-bottom: 0.8rem !important;
        }
        
        /* Input area styling */
        #chat-input {
            margin-bottom: 0.8rem !important;
        }
        
        #chat-input textarea {
            resize: vertical !important;
            min-height: 50px !important;
            max-height: 120px !important;
        }
        
        /* Button row styling */
        .button-row {
            margin-bottom: 0.5rem !important;
        }
        
        /* Reduce spacing between elements */
        .block {
            margin-bottom: 0.5rem !important;
        }
        """

    # Launch the Gradio interface
    UI.launch()
