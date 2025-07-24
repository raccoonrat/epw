# epw.py
# Implementation of Expert Pathway Watermarking (EPW) for Mixture-of-Experts (MoE) LLMs
# This is a full, end-to-end implementation for generation and detection.

import torch
import hashlib
import hmac
import math
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
import transformers.models.mixtral.modeling_mixtral as modeling_mixtral
import importlib.metadata
import warnings

# Suppress a specific warning from transformers about overriding a method.
warnings.filterwarnings("ignore", category=UserWarning, message=".*Passing `model_kwargs` to `forward` is deprecated.*")


# Check if bitsandbytes is installed for quantization
try:
    importlib.metadata.version("bitsandbytes")
    from transformers import BitsAndBytesConfig
    bitsandbytes_installed = True
except importlib.metadata.PackageNotFoundError:
    bitsandbytes_installed = False
    print("bitsandbytes not found. Quantization will be disabled.")


# =======================================================================================
# 1. WATERMARKING LOGIC
# We no longer need to subclass the model. We will use hooks for generation.
# =======================================================================================

class WatermarkLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor to embed an Expert Pathway Watermark (EPW).
    It reads the expert choice for the current token from a shared state
    object, which is populated by a hook attached during generation.
    """

    def __init__(self,
                 vocab_size: int,
                 gamma: float,
                 delta: float,
                 secret_key: str,
                 generation_state: Dict):
        if not (0 < gamma < 1):
            raise ValueError(f"gamma must be in (0, 1), but got {gamma}")

        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.secret_key = secret_key.encode('utf-8')
        self.green_list_size = int(self.vocab_size * self.gamma)
        self.generation_state = generation_state

    def _get_green_list_ids(self, expert_index: int) -> torch.Tensor:
        """Generates the green list based on the expert index."""
        seed_payload = str(expert_index).encode('utf-8')
        h = hmac.new(self.secret_key, seed_payload, hashlib.sha256)
        seed = int.from_bytes(h.digest()[:8], 'big')
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(self.vocab_size, generator=generator)
        return permutation[:self.green_list_size]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Applies the watermark to the logits at each generation step."""
        # The expert_index is provided by the generation_hook via the shared state.
        expert_index = self.generation_state.get("expert_index", 0)
        green_list_ids = self._get_green_list_ids(expert_index).to(scores.device)
        scores[:, green_list_ids] += self.delta
        return scores


# =======================================================================================
# 2. DETECTOR WITH REAL PATH RECONSTRUCTION (PYTORCH HOOKS)
# =======================================================================================
class WatermarkDetector:
    """Detects the presence of an EPW watermark in a given text."""

    def __init__(self,
                 tokenizer,
                 model,
                 secret_key: str = "default_secret_key",
                 gamma: float = 0.5):
        self.tokenizer = tokenizer
        self.model = model
        self.secret_key = secret_key.encode('utf-8')
        self.gamma = gamma
        self.device = model.device
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        self.green_list_size = int(self.vocab_size * self.gamma)

    def _get_green_list_ids(self, expert_index: int) -> torch.Tensor:
        """Generates the green list, identical to the generation logic."""
        seed_payload = str(expert_index).encode('utf-8')
        h = hmac.new(self.secret_key, seed_payload, hashlib.sha256)
        seed = int.from_bytes(h.digest()[:8], 'big')
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(self.vocab_size, generator=generator)
        return permutation[:self.green_list_size]

    def detect(self, text: str, z_threshold: float = 4.0) -> Dict[str, Any]:
        """Analyzes a text to detect the EPW watermark using real path reconstruction."""
        if not text:
            return {"detected": False, "z_score": 0.0, "message": "Input text is empty."}

        tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        num_tokens = tokenized_text.shape[1]

        if num_tokens < 10:
            return {"detected": False, "z_score": 0.0, "num_tokens": num_tokens, "message": "Text too short for reliable detection."}

        # --- REAL PATH RECONSTRUCTION USING HOOKS ---
        expert_choices = []
        def detection_hook_fn(module, args, output):
            router_logits = output
            choices = torch.argmax(router_logits, dim=-1).squeeze().tolist()
            if isinstance(choices, int):
                choices = [choices]
            expert_choices.extend(choices)

        handles = []
        # Find all MoE layers to attach hooks
        for layer in self.model.model.layers:
            if hasattr(layer, 'block_sparse_moe') and isinstance(layer.block_sparse_moe, MixtralSparseMoeBlock):
                handle = layer.block_sparse_moe.gate.register_forward_hook(detection_hook_fn)
                handles.append(handle)

        if not handles:
            return {"error": "Could not find any MoE layers to attach hooks to."}

        try:
            with torch.no_grad():
                # We must pass output_router_logits=True to get the router outputs
                self.model(tokenized_text, output_router_logits=True)
        finally:
            # Important: always remove the hooks after use
            for handle in handles:
                handle.remove()

        # The hook captures choices from all layers, we take the ones from the last layer
        num_layers_with_moe = len(handles)
        last_layer_expert_path = expert_choices[-num_tokens:]

        if len(last_layer_expert_path) != num_tokens:
             return {"error": f"Mismatch between token count ({num_tokens}) and captured expert choices ({len(last_layer_expert_path)})."}

        green_token_count = 0
        for t in range(num_tokens):
            token_id = tokenized_text[0, t].item()
            expert_index = last_layer_expert_path[t]
            green_list_ids = self._get_green_list_ids(expert_index)
            if token_id in green_list_ids:
                green_token_count += 1

        expected_green_tokens = num_tokens * self.gamma
        variance = num_tokens * self.gamma * (1 - self.gamma)
        if variance == 0:
            return {"detected": False, "z_score": float('inf')}

        z_score = (green_token_count - expected_green_tokens) / math.sqrt(variance)

        return {
            "detected": z_score > z_threshold,
            "z_score": z_score,
            "num_green_tokens": green_token_count,
            "num_tokens": num_tokens,
        }

# =======================================================================================
# 3. MAIN EXECUTION BLOCK
# =======================================================================================
if __name__ == '__main__':
    print("--- Full Implementation of EPW Framework ---")

    SECRET_KEY = "a_very_secret_and_long_key_for_hmac"
    model_name = "mistralai/Mixtral-8x7B-v0.1"
    token = "hf_xxx"

    if token == "hf_xxx":
        print("WARNING: Please replace 'hf_xxx' with your Hugging Face token.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        if bitsandbytes_installed:
            print("bitsandbytes found. Loading model with 4-bit quantization.")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        else:
            print("Loading model without quantization.")

        # Load the standard model, not the subclass.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            token=token,
        )
        model.eval()
    except Exception as e:
        print(f"\nError loading model or tokenizer: {e}")
        model, tokenizer = None, None

    if tokenizer and model:
        # --- 2. Generation with REAL Watermark (Hook-based) ---
        print("\n--- 2. Generating Watermarked Text ---")

        generation_state = {"expert_index": 0}

        def generation_hook_fn(module, args, output):
            router_logits = output
            if router_logits.ndim == 3: # (batch, seq, experts)
                final_token_logits = router_logits[:, -1, :]
            else: # (seq, experts)
                final_token_logits = router_logits[-1, :]
            expert_index = torch.argmax(final_token_logits, dim=-1).item()
            generation_state["expert_index"] = expert_index

        last_moe_layer = None
        for layer in model.model.layers:
            if hasattr(layer, 'block_sparse_moe'):
                last_moe_layer = layer.block_sparse_moe

        if last_moe_layer is None:
             raise RuntimeError("Could not find any MoE layers in the model.")

        processor = WatermarkLogitsProcessor(
             vocab_size=model.config.vocab_size,
             gamma=0.5,
             delta=4.0,
             secret_key=SECRET_KEY,
             generation_state=generation_state
        )

        prompt = "In a world where algorithms whisper secrets, one discovery changed everything."
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"Prompt: '{prompt}'")

        # FIX: Temporarily disable the problematic load balancing loss during generation.
        # This is a regression fix.
        handle = last_moe_layer.gate.register_forward_hook(generation_hook_fn)
        original_load_balancing_loss_func = modeling_mixtral.load_balancing_loss_func
        modeling_mixtral.load_balancing_loss_func = lambda *a, **kw: torch.tensor(0.0, device=model.device)

        try:
            output_watermarked = model.generate(
                **input_ids,
                max_new_tokens=50,
                logits_processor=[processor],
                do_sample=True,
                top_k=0,
                temperature=0.7,
                output_router_logits=True # Crucial to trigger the hook's `output`
            )
        finally:
            # Always remove the hook and restore the original function.
            handle.remove()
            modeling_mixtral.load_balancing_loss_func = original_load_balancing_loss_func

        watermarked_text = tokenizer.decode(output_watermarked[0], skip_special_tokens=True)

        # Generate unwatermarked text for comparison (no special handling needed).
        output_unwatermarked = model.generate(
            **input_ids, max_new_tokens=50, do_sample=True, top_k=0, temperature=0.7
        )
        unwatermarked_text = tokenizer.decode(output_unwatermarked[0], skip_special_tokens=True)

        print(f"\nWatermarked Text:\n{watermarked_text}")
        print(f"\nUnwatermarked Text:\n{unwatermarked_text}")

        # --- 3. Detection with REAL Path Reconstruction ---
        print("\n--- 3. Detecting Watermark ---")
        detector = WatermarkDetector(
            tokenizer=tokenizer, model=model, secret_key=SECRET_KEY, gamma=0.5
        )

        generated_watermarked_text = watermarked_text[len(prompt):]
        generated_unwatermarked_text = unwatermarked_text[len(prompt):]

        print("\nAnalyzing watermarked text...")
        result_watermarked = detector.detect(generated_watermarked_text)
        print(f"--> Detection Result: {result_watermarked}")
        if result_watermarked.get("detected"):
            print("--> Verdict: Watermark DETECTED (Z-score is high as expected).")
        else:
            print("--> Verdict: Watermark NOT detected (Z-score is low).")

        print("\nAnalyzing unwatermarked text...")
        result_unwatermarked = detector.detect(generated_unwatermarked_text)
        print(f"--> Detection Result: {result_unwatermarked}")
        if not result_unwatermarked.get("detected"):
            print("--> Verdict: Watermark NOT detected (Z-score is low as expected).")
        else:
            print("--> Verdict: Watermark DETECTED (This would be a false positive).")

    else:
        print("\nSkipping demonstration due to model or tokenizer loading error.")
