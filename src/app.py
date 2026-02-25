import gradio as gr

def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="👋 Greeting Demo",
    description="Enter your name to receive a warm greeting."
)

if __name__ == "__main__":
    demo.launch()
