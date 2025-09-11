from src.qwen2_v2 import my_qwen2_cpu as my_qwen2
import gradio as gr

dropdown_options = my_qwen2.SUPPORT_MODELS


def generate_response(model_name, user_input):
    model_name = "Qwen/" + model_name
    model = my_qwen2.Qwen2(model_name, max_new_tokens=100)
    response = model.generate(user_input)
    return response


if __name__ == "__main__":
    UI = gr.Interface(
        fn=generate_response,
        inputs=[
            gr.Dropdown(choices=dropdown_options, label="Select an model to chat"),
            gr.Textbox(lines=5, placeholder="type your question here..."),
        ],
        outputs="text",
        title=" demo",
        description="This is a demo for your own AI agent.",
        theme=gr.themes.Monochrome(),
    )

    UI.launch()
