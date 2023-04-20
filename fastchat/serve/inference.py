"""Inference for FastChat models."""
import abc
from typing import Optional
import warnings

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, AutoModel, \
        LlamaForCausalLM
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LLaMATokenizer, LLamaForCausalLM, AutoModel

from fastchat.conversation import conv_templates, get_default_conv_template, SeparatorStyle
from fastchat.serve.compression import compress_module
from fastchat.serve.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations
from fastchat.serve.serve_chatglm import chatglm_generate_stream


def raise_warning_for_old_weights(model_path, model):
    if "vicuna" in model_path.lower():
        try:
            is_vicuna = isinstance(model, LlamaForCausalLM)
        except Exception:
            is_vicuna = isinstance(model, LLamaForCausalLM)
        if is_vicuna and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fschat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.1: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template conv_one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n")


def compute_skip_echo_len(model_name, conv, prompt):
    model_name = model_name.lower()
    if "chatglm" in model_name:
        skip_echo_len = len(conv.messages[-2][1]) + 1
    elif "dolly" in model_name:
        special_toks = ["### Instruction:", "### Response:", "### End"]
        prompt_tmp = prompt
        for tok in special_toks:
            prompt_tmp = prompt_tmp.replace(tok, "")
        skip_echo_len = len(prompt_tmp)
    else:
        skip_echo_len = len(prompt) + 1 - prompt.count("</s>") * 3
    return skip_echo_len


def load_model(model_path, device, num_gpus, max_gpu_memory="13GiB",
               load_8bit=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: max_gpu_memory for i in range(num_gpus)},
                })
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    elif "dolly" in model_path:
        kwargs.update({"torch_dtype": torch.bfloat16})
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     low_cpu_mem_usage=True, **kwargs)
        raise_warning_for_old_weights(model_path, model)

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


@torch.inference_mode()
def generate_stream(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))

    # TODO ADD top_k top_p parameter
    # top_k = int(params.get("top_k", 50))
    # top_p = float(params.get("top_p", 1.0))
    # assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    # assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."

    stop_str = params.get("stop", None)
    if stop_str == tokenizer.eos_token:
        stop_str = None

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
            yield output

        if stopped:
            break

    del past_key_values


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)


def generate_prompt(instruction):
    return PROMPT_TEMPLATE.format_map({'instruction': instruction})


# @torch.inference_mode()
def generate_base(model, tokenizer, params, device,
                  context_len=2048):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    template = params.get("template", None)
    temperature = float(params.get("temperature", 0.9))
    max_new_tokens = int(params.get("max_new_tokens", 4000))
    top_k = int(params.get("top_k", 50))
    top_p = float(params.get("top_p", 1.0))
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."

    generation_config = dict(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.0,
    )

    stop_str = params.get("stop", None)
    if stop_str == tokenizer.eos_token:
        stop_str = None

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    if model_vocab_size != tokenizer_vocab_size:
        assert tokenizer_vocab_size > model_vocab_size
        model.resize_token_embeddings(tokenizer_vocab_size)

    model.eval()

    with torch.no_grad():
        raw_input_text = prompt
        if template == "normal":
            input_text = generate_prompt(instruction=raw_input_text)
        else:
            input_text = raw_input_text
        try:
            inputs = tokenizer(input_text, return_tensors="pt")
        except Exception as e:
            return f"tokenizer error: {e}"
        generation_output = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs['attention_mask'].cuda(),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_config
        )
        s = generation_output[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        if template is not None:
            output = output.split("### Response:")[1].strip()
        else:
            # remote prompt
            output = output[l_prompt:]
        return output


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream, skip_echo_len: int):
        """Stream output."""


def chat_loop(model_path: str, device: str, num_gpus: str,
              max_gpu_memory: str, load_8bit: bool,
              conv_template: Optional[str], temperature: float,
              max_new_tokens: int, chatio: ChatIO,
              debug: bool):
    # Model
    model, tokenizer = load_model(model_path, device,
                                  num_gpus, max_gpu_memory, load_8bit, debug)
    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = get_default_conv_template(model_path).copy()

    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            prompt = conv.messages[conv.offset:]
            generate_stream_func = chatglm_generate_stream
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        skip_echo_len = compute_skip_echo_len(model_path, conv, prompt)

        params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, params, device)
        outputs = chatio.stream_output(output_stream, skip_echo_len)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
