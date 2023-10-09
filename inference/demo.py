# Load the model.
# Note: It can take a while to download LLaMA and add the adapter modules.

import torch
import fire

import datetime
import os
from threading import Event, Thread
from uuid import uuid4

import gradio as gr
import requests

from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    LlamaTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

torch.backends.cuda.matmul.allow_tf32 = True


def main(
    model_name: str = "/path/to/llama-2-70b-hf",
    adapters_name: str = "/path/to/adapter_model",
    tokenizer_name: str = "TheBloke/dromedary-65b-lora-HF",  # a random loadable llama tokenizer
    max_queue_size: int = 16,
    sharable_link: bool = True,
):
    print(f"Starting to load the model {model_name} into memory")

    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    m = PeftModel.from_pretrained(m, adapters_name, is_trainable=False)

    start_message = (
        "# Dromedary"
        "\n\n## System Overview"
        "\n\nConsider an AI assistant whose codename is Dromedary, developed by the Self-Align team. "
        "Dromedary is trained on data from before Sept-2022, and it endeavors to be a helpful, ethical and reliable assistant."
        "\n\n## User Conversation\n\n"
    )

    tok = LlamaTokenizerFast.from_pretrained(tokenizer_name)

    print(f"Successfully loaded the model {model_name} into memory")

    # Setup the gradio Demo.

    stop_tokens = ["### User"]

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            decoded_tokens = tok.batch_decode(input_ids, skip_special_tokens=True)

            for stop_token in stop_tokens:
                if decoded_tokens[0].strip().endswith(stop_token):
                    return True
            return False

    def convert_history_to_text(history):
        text = start_message + "".join(
            [
                "".join(
                    [
                        f"### User\n{item[0]}\n\n",
                        f"### Dromedary\n{item[1]}\n\n",
                    ]
                )
                for item in history[:-1]
            ]
        )
        text += "".join(
            [
                "".join(
                    [
                        f"### User\n{history[-1][0]}\n\n",
                        f"### Dromedary\n{history[-1][1]}",
                    ]
                )
            ]
        )
        return text

    def log_conversation(conversation_id, history, messages, generate_kwargs):
        logging_url = os.getenv("LOGGING_URL", None)
        if logging_url is None:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        data = {
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "history": history,
            "messages": messages,
            "generate_kwargs": generate_kwargs,
        }

        try:
            requests.post(logging_url, json=data)
        except requests.exceptions.RequestException as e:
            print(f"Error logging conversation: {e}")

    def user(message, history):
        # Append the user's message to the conversation history
        return "", history + [[message, ""]]

    def bot(
        history, temperature, top_p, max_new_tokens, repetition_penalty, conversation_id
    ):
        # Initialize a StopOnTokens object
        stop = StopOnTokens()

        # Construct the input message string for the model by concatenating the current system message and conversation history
        messages = convert_history_to_text(history)

        print(f"history:")
        print(messages)

        # Tokenize the messages string
        input_ids = tok(messages, return_tensors="pt").input_ids
        input_ids = input_ids.to(m.device)
        streamer = TextIteratorStreamer(
            tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([stop]),
        )

        stream_complete = Event()

        def generate_and_signal_complete():
            m.generate(**generate_kwargs)
            stream_complete.set()

        def log_after_stream_complete():
            stream_complete.wait()
            log_conversation(
                conversation_id,
                history,
                messages,
                {
                    "max_new_tokens": max_new_tokens,
                    "top_p": top_p,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                },
            )

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        t2 = Thread(target=log_after_stream_complete)
        t2.start()

        # Initialize an empty string to store the generated text
        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            history[-1][1] = partial_text.split("### User")[0]
            yield history

        print(f"output:")
        print(history[-1][1])

    def get_uuid():
        return str(uuid4())

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        conversation_id = gr.State(get_uuid)
        gr.Markdown(
            """Dromedary-2 Demo
    """
        )
        chatbot = gr.Chatbot().style(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                ).style(container=False)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    # stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
        with gr.Row():
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.7,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=0.9,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            max_new_tokens = gr.Slider(
                                label="Response Length",
                                value=768,
                                minimum=64,
                                maximum=1024,
                                step=64,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.0,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
        with gr.Row():
            gr.Markdown(
                "Disclaimer: The model can produce factually incorrect output, and should not be relied on to produce "
                "factually accurate information. The model was trained on various public datasets; while great efforts "
                "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
                "biased, or otherwise offensive outputs.",
                elem_classes=["disclaimer"],
            )
        with gr.Row():
            gr.Markdown(
                "[Privacy policy](https://gist.github.com/samhavens/c29c68cdcd420a9aa0202d0839876dac)",
                elem_classes=["disclaimer"],
            )

        submit_event = msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                temperature,
                top_p,
                max_new_tokens,
                repetition_penalty,
                conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        submit_click_event = submit.click(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                temperature,
                top_p,
                max_new_tokens,
                repetition_penalty,
                conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        # stop.click(
        #     fn=None,
        #     inputs=None,
        #     outputs=None,
        #     cancels=[submit_event, submit_click_event],
        #     queue=False,
        # )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue(max_size=max_queue_size, concurrency_count=1)

    # Launch your Dromedary-2 Demo!
    demo.launch(share=sharable_link)


if __name__ == "__main__":
    fire.Fire(main)
