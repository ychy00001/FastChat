import argparse
from collections import defaultdict
import datetime
import json
import os
import time
import uuid

import gradio as gr
import requests

from fastchat.conversation import (get_default_conv_template,
                                   SeparatorStyle)
from fastchat.constants import LOGDIR
from fastchat.utils import (build_logger, server_error_msg,
                            violates_moderation, moderation_msg)
from fastchat.serve.gradio_patch import Chatbot as grChatbot
from fastchat.serve.gradio_css import code_highlight_css
from fastchat.serve.inference import compute_skip_echo_len

from fastapi import FastAPI

logger = build_logger("gradio_web_server", "/data/project/FastChat/log/gradio_web_server.log")

headers = {"User-Agent": "fastchat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
    "dolly-v2-12b": "aaaaaac",
    "chatglm-6b": "aaaaaad",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"/data/project/FastChat/log/{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(
                value=model, visible=True)

    state = None
    return (state,
            dropdown_update,
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True))


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = None
    return (state, gr.Dropdown.update(
        choices=models,
        value=models[0] if len(models) > 0 else ""),
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True))


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a", encoding='utf-8') as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    return (state, [], "") + (disable_btn,) * 5


def add_text(state, text, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")

    if state is None:
        state = get_default_conv_template("vicuna").copy()

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def http_bot(state, model_selector, temperature, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        new_state = get_default_conv_template(model_name).copy()
        new_state.conv_id = uuid.uuid4().hex
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
                        json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    if "chatglm" in model_name:
        prompt = state.messages[state.offset:]
    else:
        prompt = state.get_prompt()
    skip_echo_len = compute_skip_echo_len(model_name, state, prompt)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2,
    }
    logger.info(f"==== request ====\n{pload}")

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
                                 headers=headers, json=pload, stream=True, timeout=20)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][skip_echo_len:].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.02)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg + f" (error_code: 4)"
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")


notice_markdown = ("""
#  云从数据研究院
大语言模型功能内测
""")

learn_more_markdown = ("""
### 问题反馈
有任何疑问，请咨询数研院。
""")

css = code_highlight_css + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""


def build_demo():
    with gr.Blocks(title="FastChat", theme=gr.themes.Base(), css=css) as demo:
        state = gr.State()
        # Draw layout
        notice = gr.Markdown(notice_markdown)

        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False).style(container=False)

        chatbot = grChatbot(elem_id="chatbot", visible=False, label="cw_chat").style(height=550)
        with gr.Row():
            with gr.Column(scale=20):
                textbox = gr.Textbox(show_label=False,
                                     placeholder="请输入内容，回车发送", visible=False).style(container=False)
            with gr.Column(scale=1, min_width=150):
                submit_btn = gr.Button(value="发送", visible=False)

        with gr.Row(visible=False) as button_row:
            upvote_btn = gr.Button(value="✔️  赞一个", interactive=False, variant="primary").style(size='lg')
            downvote_btn = gr.Button(value="✖️  踩一脚", interactive=False, variant="primary").style(size='lg')
            flag_btn = gr.Button(value="📌️  标记一下", interactive=False)
            # stop_btn = gr.Button(value="⏹  Stop Generation", interactive=False)
            regenerate_btn = gr.Button(value="🔄  重新生成", interactive=False)
            clear_btn = gr.Button(value="🚮  清除会话", interactive=False)

        with gr.Accordion("参数配置", open=False, visible=False) as parameter_row:
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True,
                                    label="Temperature", )
            max_output_tokens = gr.Slider(minimum=0, maximum=4096, value=128, step=64, interactive=True,
                                          label="Max output tokens", )

        # gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(upvote_last_response,
                         [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        downvote_btn.click(downvote_last_response,
                           [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        flag_btn.click(flag_last_response,
                       [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        regenerate_btn.click(regenerate, state,
                             [state, chatbot, textbox] + btn_list).then(
            http_bot, [state, model_selector, temperature, max_output_tokens],
            [state, chatbot] + btn_list)
        clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

        model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

        textbox.submit(add_text, [state, textbox], [state, chatbot, textbox] + btn_list
                       ).then(http_bot, [state, model_selector, temperature, max_output_tokens],
                              [state, chatbot] + btn_list)
        submit_btn.click(add_text, [state, textbox], [state, chatbot, textbox] + btn_list
                         ).then(http_bot, [state, model_selector, temperature, max_output_tokens],
                                [state, chatbot] + btn_list)

        if args.model_list_mode == "once":
            demo.load(load_demo, [url_params], [state, model_selector,
                                                chatbot, textbox, submit_btn, button_row, parameter_row],
                      _js=get_window_url_params)
        elif args.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None, [state, model_selector,
                                                           chatbot, textbox, submit_btn, button_row, parameter_row])
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


invite_code_list_file = "/data/project/FastChat/passwd"
invite_code_list = []
f = open(invite_code_list_file)  # 返回一个文件对象
line = f.readline()  # 调用文件的 readline()方法
while line:
    if line.isspace():
        continue
    invite_code_list.append(line.strip())
    line = f.readline()
f.close()


def simple_auth(username, password):
    # 记录登录日志
    logger.info(f"##login## : {password}")
    return password in invite_code_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once",
                        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true",
                        help="Enable content moderation")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo()
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10,
               api_open=False).launch(server_name=args.host, server_port=args.port,
                                      share=args.share, max_threads=200, auth=simple_auth, auth_message="请输入您的邀请账号和密码")
