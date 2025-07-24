# epw.py
# Implementation of Expert Pathway Watermarking (EPW) for Mixture-of-Experts (MoE) LLMs
# This is a full, end-to-end implementation for generation and detection.

import torch
import hashlib
import hmac
import math
from typing import List, Optional, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralSparseMoeBlock
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
# 1. MODIFIED MODEL CLASS TO EXPOSE ROUTER LOGITS
# =======================================================================================
class MixtralForCausalLMWithWatermark(MixtralForCausalLM):
    """
    A subclass of MixtralForCausalLM that is modified to return router logits.
    This makes the expert choices accessible for the watermarking processor.
    """
    def forward(self, *args, **kwargs):
        # FIX: A more robust fix for the internal load balancing loss error.
        # We temporarily monkey-patch the problematic function with a dummy that
        # returns a zero tensor. This is safe during inference as the loss is not used.
        original_load_balancing_loss_func = modeling_mixtral.load_balancing_loss_func
        modeling_mixtral.load_balancing_loss_func = lambda *a, **kw: torch.tensor(0.0, device=self.device)
        
        try:
            # We set output_router_logits=True to ensure the router decisions are in the output.
            kwargs['output_router_logits'] = True
            outputs = super().forward(*args, **kwargs)
        finally:
            # Always restore the original function to not affect other operations.
            modeling_mixtral.load_balancing_loss_func = original_load_balancing_loss_func

        if outputs.router_logits:
            # Store the raw router logits from the last layer. The processor will
            # be responsible for slicing it correctly based on the context.
            self.last_router_logits = outputs.router_logits[-1]
        else:
            self.last_router_logits = None
            
        return outputs

class WatermarkLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor to embed an Expert Pathway Watermark (EPW).
    """

    def __init__(self,
                 model: MixtralForCausalLMWithWatermark,
                 vocab_size: int,
                 gamma: float = 0.5,
                 delta: float = 2.0,
                 secret_key: str = "default_secret_key"):
        if not (0 < gamma < 1):
            raise ValueError(f"gamma must be in (0, 1), but got {gamma}")

        self.model = model
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.secret_key = secret_key.encode('utf-8')
        self.green_list_size = int(self.vocab_size * self.gamma)

    def _get_green_list_ids(self, expert_index: int) -> torch.Tensor:
        seed_payload = str(expert_index).encode('utf-8')
        h = hmac.new(self.secret_key, seed_payload, hashlib.sha256)
        seed = int.from_bytes(h.digest()[:8], 'big')
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(self.vocab_size, generator=generator)
        return permutation[:self.green_list_size]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Applies the watermark to the logits at each generation step.
        """
        if getattr(self.model, 'last_router_logits', None) is None:
            return scores
        
        raw_router_logits = self.model.last_router_logits

        # This logic handles the various tensor shapes that can occur during generation.
        if raw_router_logits.ndim == 3:
            # Case 1: Batch dim exists. Shape: (batch, seq_len, experts).
            final_token_logits = raw_router_logits[:, -1, :]
        else: # ndim == 2
            # Case 2: Batch dim was squeezed. Shape: (seq_len, experts).
            final_token_logits = raw_router_logits[-1, :]

        # final_token_logits is now shape (batch, experts) or (experts).
        expert_index = torch.argmax(final_token_logits, dim=-1).item()

        green_list_ids = self._get_green_list_ids(expert_index).to(scores.device)
        scores[:, green_list_ids] += self.delta
        return scores


# =======================================================================================
# 2. DETECTOR WITH REAL PATH RECONSTRUCTION (PYTORCH HOOKS)
# =======================================================================================
class WatermarkDetector:
    """
    Detects the presence of an Expert Pathway Watermark in a given text.
    """

    def __init__(self,
                 tokenizer,
                 model,
                 secret_key: str = "default_secret_key",
                 gamma: float = 0.5,
                 device: Optional[str] = None):
        self.tokenizer = tokenizer
        self.model = model
        self.secret_key = secret_key.encode('utf-8')
        self.gamma = gamma
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        self.green_list_size = int(self.vocab_size * self.gamma)

    def _get_green_list_ids(self, expert_index: int) -> torch.Tensor:
        seed_payload = str(expert_index).encode('utf-8')
        h = hmac.new(self.secret_key, seed_payload, hashlib.sha256)
        seed = int.from_bytes(h.digest()[:8], 'big')
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(self.vocab_size, generator=generator)
        return permutation[:self.green_list_size]


    def detect(self, text: str, z_threshold: float = 4.0) -> Dict[str, Any]:
        if not text:
            return {"detected": False, "z_score": 0.0, "message": "Input text is empty."}
            
        tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        num_tokens = tokenized_text.shape[1]

        if num_tokens < 10:
            return {"detected": False, "z_score": 0.0, "num_tokens": num_tokens, "message": "Text too short for reliable detection."}

        # --- REAL PATH RECONSTRUCTION USING HOOKS ---
        expert_choices = []
        def hook_fn(module, args, output):
            # =========================================================================
            # FIX: The output of the gate layer is the tensor itself, not a tuple.
            # This was the cause of the token/choice mismatch error.
            # =========================================================================
            router_logits = output
            choices = torch.argmax(router_logits, dim=-1).squeeze().tolist()
            if isinstance(choices, int):
                choices = [choices]
            expert_choices.extend(choices)

        handles = []
        for layer in self.model.model.layers:
            if isinstance(layer.block_sparse_moe, MixtralSparseMoeBlock):
                handle = layer.block_sparse_moe.gate.register_forward_hook(hook_fn)
                handles.append(handle)
        
        if not handles:
            return {"error": "Could not find any MoE layers to attach hooks to."}

        try:
            with torch.no_grad():
                self.model(tokenized_text)
        finally:
            for handle in handles:
                handle.remove()

        # The hooks are called for each MoE layer. We only need the path from one layer.
        # Here, we use the path from the last MoE layer.
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

    # --- 1. Setup ---
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

        model = MixtralForCausalLMWithWatermark.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            token=token,
        )
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"\nError loading model or tokenizer: {e}")
        print("This may be due to an invalid token, network issues, or insufficient memory.")
        model, tokenizer = None, None

    if tokenizer and model:
        # --- 2. Generation with REAL Watermark ---
        print("\n--- 2. Generating Watermarked Text ---")
        processor = WatermarkLogitsProcessor(
             model=model,
             vocab_size=model.config.vocab_size,
             gamma=0.5,
             delta=4.0,
             secret_key=SECRET_KEY,
        )

        prompt = "In a world where algorithms whisper secrets, one discovery changed everything."
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

        print(f"Prompt: '{prompt}'")
        
        output_watermarked = model.generate(
            **input_ids,
            max_new_tokens=50,
            logits_processor=[processor],
            do_sample=True,
            top_k=0,
            temperature=0.7,
        )
        watermarked_text = tokenizer.decode(output_watermarked[0], skip_special_tokens=True)
        
        output_unwatermarked = model.generate(
            **input_ids,
            max_new_tokens=50,
            do_sample=True,
            top_k=0,
            temperature=0.7,
        )
        unwatermarked_text = tokenizer.decode(output_unwatermarked[0], skip_special_tokens=True)
        
        print(f"\nWatermarked Text:\n{watermarked_text}")
        print(f"\nUnwatermarked Text:\n{unwatermarked_text}")

        # --- 3. Detection with REAL Path Reconstruction ---
        print("\n--- 3. Detecting Watermark ---")
        detector = WatermarkDetector(
            tokenizer=tokenizer,
            model=model,
            secret_key=SECRET_KEY,
            gamma=0.5
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

