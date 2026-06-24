"""
Unified multi-VLM registry + adapter for maritime evaluation.

Uses transformers 5.x AutoModelForImageTextToText + AutoProcessor chat templates
to give a single generate() interface across heterogeneous VLM families, so the
maritime zero-shot benchmark (hallucination / COLREG reasoning / grounding) runs
identically on every model. Gated models require an HF token + license accept.

Sizes chosen to fit a single 32GB GPU (bf16; 4-bit via bitsandbytes if available).
"""
import torch
from PIL import Image

# name -> (hf_repo, gated, notes). Sizes picked to fit 32GB single GPU.
REGISTRY = {
    "qwen2-vl":     ("Qwen/Qwen2-VL-7B-Instruct",            False, "custom ViT"),
    "qwen2.5-vl":   ("Qwen/Qwen2.5-VL-7B-Instruct",          False, "custom ViT"),
    "llava-1.5":    ("llava-hf/llava-1.5-7b-hf",             False, "CLIP-ViT-L"),
    "llava-next":   ("llava-hf/llava-v1.6-mistral-7b-hf",    False, "CLIP-ViT-L"),
    "internvl3":    ("OpenGVLab/InternVL3-8B-hf",            False, "InternViT"),
    "idefics2":     ("HuggingFaceM4/idefics2-8b",            False, "SigLIP"),
    "smolvlm":      ("HuggingFaceTB/SmolVLM-Instruct",       False, "SigLIP"),
    "paligemma":    ("google/paligemma2-3b-mix-448",         True,  "SigLIP (gated)"),
    "llama-3.2":    ("meta-llama/Llama-3.2-11B-Vision-Instruct", True, "cross-attn (gated)"),
    "aya-vision":   ("CohereForAI/aya-vision-8b",            True,  "SigLIP (gated)"),
    "gemma-3":      ("google/gemma-3-4b-it",                 True,  "SigLIP (gated)"),
}


def load(name, dtype=torch.bfloat16, four_bit=False):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    repo = REGISTRY[name][0]
    kw = dict(torch_dtype=dtype, device_map="cuda")
    if four_bit:
        try:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
            kw.pop("torch_dtype", None)
        except Exception as e:
            print(f"  [zoo] 4bit unavailable ({e}); using bf16")
    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(repo, **kw).eval()
    return model, processor


@torch.no_grad()
def generate(model, processor, image, prompt, max_new_tokens=128):
    """Uniform chat-template generation. image: PIL.Image. Returns decoded new text."""
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    inputs = None
    # 1) chat-template models (most VLMs)
    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", images=[image])
    except TypeError:
        try:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, images=image, return_tensors="pt")
        except Exception:
            inputs = None
    except Exception:
        inputs = None
    # 2) non-chat models (e.g. PaliGemma): plain prompt + image
    if inputs is None:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][in_len:] if in_len else out[0]
    return processor.decode(gen, skip_special_tokens=True).strip()
