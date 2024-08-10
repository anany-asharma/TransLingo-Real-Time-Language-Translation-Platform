# importing required libraries
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr


# creating model
# T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format.
model_name = "google-t5/t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# creating translation function
def translate(lan2, text):
    text = "translate English to " + lan2 + ": " + text
    input = tokenizer(text, return_tensors="pt")
    output = model.generate(**input, max_length=20)
    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    return translation


# creating interface
iface = gr.Interface(
    fn=translate,
    inputs=[gr.Dropdown(["French","German","Romanian"],label='Translate to',info="Select a language"),gr.Textbox(lines=2,placeholder='Text to translate')],
    outputs='text',
    title="TransLingo",
)


iface.launch(share=True)