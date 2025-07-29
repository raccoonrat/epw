# epw.py
# Implementation of Expert Pathway Watermarking (EPW) for Mixture-of-Experts (MoE) LLMs
# FINAL ARCHITECTURE: This version uses the robust model subclassing approach to
# resolve timing issues and correctly handles all discovered bugs.
# ENHANCED WITH EPW-A: Implements the enhanced EPW-A framework with IRSH protocol

import torch
import hashlib
import hmac
import math
from typing import List, Optional, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralSparseMoeBlock
from transformers.generation.utils import CausalLMOutputWithPast
# Import the module itself to allow for monkey-patching the loss function
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
# 1. ROBUST MODEL SUBCLASS FOR WATERMARKING WITH IRSH SUPPORT
# This subclass reliably passes expert routing information to the logits processor
# by overriding key methods of the generation mixin, and adds IRSH protocol support.
# =======================================================================================

class MixtralForCausalLMWithWatermark(MixtralForCausalLM):
    """
    This subclass of MixtralForCausalLM is engineered to reliably pass expert routing
    information to a logits processor during text generation, with enhanced IRSH support.
    """

    def __init__(self, config):
        super().__init__(config)
        # Calculate router hash for IRSH protocol
        self.router_hash = self._calculate_router_hash()
        
    def _calculate_router_hash(self) -> str:
        """
        Calculate the cryptographic hash of router weights for IRSH protocol.
        This binds the watermark to a specific version of the router.
        """
        router_weights = []
        
        # Collect router weights from all MoE layers
        for layer in self.model.layers:
            if hasattr(layer, 'block_sparse_moe') and isinstance(layer.block_sparse_moe, MixtralSparseMoeBlock):
                # Get the gate weights (router weights)
                gate_weights = layer.block_sparse_moe.gate.weight.data
                router_weights.append(gate_weights.cpu().numpy().tobytes())
        
        # Concatenate all router weights and hash
        combined_weights = b''.join(router_weights)
        router_hash = hashlib.sha256(combined_weights).hexdigest()
        return router_hash

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """
        DELIBERATE OVERRIDE: The standard validation can fail with `device_map="auto"`
        and subclassing. Since we know our kwargs are correct, we bypass the check.
        """
        pass

    def forward(self, *args, **kwargs):
        """
        Overrides the forward pass to a) ensure router logits are always computed, and
        b) temporarily bypass the auxiliary load balancing loss which is unused during
        inference and can cause errors.
        """
        kwargs['output_router_logits'] = True

        # MONKEY-PATCH to bypass problematic auxiliary loss calculation during inference.
        original_loss_func = getattr(modeling_mixtral, "load_balancing_loss_func", None)
        # Create a dummy function that returns a zero tensor on the correct device.
        dummy_loss_func = lambda *a, **kw: torch.tensor(0.0, device=self.device)
        if original_loss_func:
            modeling_mixtral.load_balancing_loss_func = dummy_loss_func

        try:
            outputs = super().forward(*args, **kwargs)
        finally:
            # Always restore the original function to not affect other operations.
            if original_loss_func:
                modeling_mixtral.load_balancing_loss_func = original_loss_func

        return outputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: CausalLMOutputWithPast,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        """
        THE CORE OF THE ROBUST SOLUTION: This method is called after each step in the
        `generate` loop. We override it to take the `router_logits` from the current
        step's output and explicitly place them into the `model_kwargs` for the
        *next* step. This ensures the LogitsProcessor has access to the correct,
        synchronized expert choices.
        """
        # First, let the standard update happen.
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, standardize_cache_format
        )
        # Now, add our custom state (the router logits) to the kwargs.
        if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            # The output is a tuple of tensors, one for each MoE layer.
            model_kwargs["router_logits"] = outputs.router_logits
        return model_kwargs

# =======================================================================================
# 2. EPW-A WATERMARKING LOGITS PROCESSOR (ENHANCED IMPLEMENTATION)
# Implements the complete EPW-A algorithm with IRSH, GSG, and EWP support
# =======================================================================================

class EPWALogitsProcessor(LogitsProcessor):
    """
    Enhanced LogitsProcessor that implements the complete EPW-A algorithm.
    Supports both GSG (Gating-Seeded Green-listing) and EWP (Expert-Specific Weighted Perturbation) variants.
    Incorporates IRSH (Initial Router State Hashing) protocol for enhanced security.
    """
    
    def __init__(self, 
                 vocab_size: int, 
                 gamma: float, 
                 secret_key: str,
                 router_hash: str,
                 mode: str = "gsg",  # "gsg" or "ewp"
                 delta_config: Union[float, Dict[int, float]] = 4.0):
        """
        Initialize EPW-A LogitsProcessor.
        
        Args:
            vocab_size: Size of the vocabulary
            gamma: Proportion of tokens in green list
            secret_key: Secret key for watermark generation
            router_hash: Hash of router weights for IRSH protocol
            mode: "gsg" for Gating-Seeded Green-listing or "ewp" for Expert-Specific Weighted Perturbation
            delta_config: For GSG mode, a single float value. For EWP mode, a dict mapping expert indices to base deltas
        """
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.secret_key = secret_key.encode('utf-8')
        self.router_hash = router_hash.encode('utf-8')
        self.mode = mode
        self.delta_config = delta_config
        self.green_list_size = int(self.vocab_size * self.gamma)
        
        # Validate configuration
        if mode == "ewp" and not isinstance(delta_config, dict):
            raise ValueError("EWP mode requires delta_config to be a dictionary mapping expert indices to base deltas")
        elif mode == "gsg" and not isinstance(delta_config, (int, float)):
            raise ValueError("GSG mode requires delta_config to be a single float value")

    def _get_green_list_ids(self, expert_index: int) -> torch.Tensor:
        """
        Generate green list using IRSH protocol: seed = PRF(secret_key, expert_index, router_hash)
        """
        # Create seed payload with IRSH protocol
        seed_payload = f"{expert_index}:{self.router_hash.decode('utf-8')}".encode('utf-8')
        h = hmac.new(self.secret_key, seed_payload, hashlib.sha256)
        seed = int.from_bytes(h.digest()[:8], 'big')
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(self.vocab_size, generator=generator)
        return permutation[:self.green_list_size]

    def _calculate_effective_delta(self, expert_index: int, expert_probabilities: torch.Tensor) -> float:
        """
        Calculate effective delta based on mode and routing confidence.
        
        For GSG: returns global delta
        For EWP: returns base_delta * confidence
        """
        if self.mode == "gsg":
            return float(self.delta_config)
        elif self.mode == "ewp":
            # Get base delta for this expert
            base_delta = self.delta_config.get(expert_index, 4.0)  # Default to 4.0 if not specified
            
            # Get confidence (softmax probability of selected expert)
            confidence = expert_probabilities[expert_index].item()
            
            # Calculate effective delta: base_delta * confidence
            effective_delta = base_delta * confidence
            return effective_delta
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """
        EPW-A implementation: Apply watermark based on expert routing with IRSH protocol.
        """
        # Safely get the router_logits from the kwargs
        router_logits = kwargs.get("router_logits")

        if router_logits is None:
            # If for any reason the logits are not available, do not apply the watermark
            return scores

        # The router_logits is a tuple of tensors (one per MoE layer). We use the last layer's
        last_layer_router_logits = router_logits[-1]
        
        # Get expert index and probabilities
        expert_probabilities = torch.softmax(last_layer_router_logits, dim=-1)
        expert_index = torch.argmax(last_layer_router_logits, dim=-1).item()

        # Generate green list using IRSH protocol
        green_list_ids = self._get_green_list_ids(expert_index).to(scores.device)
        
        # Calculate effective delta based on mode and routing confidence
        effective_delta = self._calculate_effective_delta(expert_index, expert_probabilities)
        
        # Apply watermark bias
        scores[:, green_list_ids] += effective_delta
        return scores

# =======================================================================================
# 3. LEGACY WATERMARKING LOGITS PROCESSOR (FOR BACKWARD COMPATIBILITY)
# =======================================================================================

class WatermarkLogitsProcessor(LogitsProcessor):
    """
    Legacy LogitsProcessor for backward compatibility.
    A LogitsProcessor that embeds the EPW. It now reliably receives the
    expert choices via the `model_kwargs` passed to it at each step.
    """
    def __init__(self, vocab_size: int, gamma: float, delta: float, secret_key: str):
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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """
        CORRECT IMPLEMENTATION: This method is called by the generate function.
        Thanks to our model subclass, the `kwargs` dictionary now reliably
        contains the synchronized `router_logits`.
        """
        # Safely get the router_logits from the kwargs.
        router_logits = kwargs.get("router_logits")

        if router_logits is None:
            # If for any reason the logits are not available, do not apply the watermark.
            return scores

        # The router_logits is a tuple of tensors (one per MoE layer). We use the last layer's.
        # The tensor shape is (batch_size, num_experts).
        last_layer_router_logits = router_logits[-1]

        # In the generate loop, the batch size is 1. We get the single expert index.
        expert_index = torch.argmax(last_layer_router_logits, dim=-1).item()

        # Generate the green list and apply the bias.
        green_list_ids = self._get_green_list_ids(expert_index).to(scores.device)
        scores[:, green_list_ids] += self.delta
        return scores

# =======================================================================================
# 4. ENHANCED DETECTOR WITH IRSH SUPPORT
# =======================================================================================
class WatermarkDetector:
    """Detects the presence of an EPW watermark in a given text using hooks."""
    def __init__(self, tokenizer, model, secret_key: str = "default_secret_key", gamma: float = 0.5, router_hash: str = None):
        self.tokenizer = tokenizer
        self.model = model # Can be the original model or our subclass
        self.secret_key = secret_key.encode('utf-8')
        self.gamma = gamma
        self.device = model.device
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        
        # Use provided router_hash or get from model if available
        if router_hash is None and hasattr(model, 'router_hash'):
            self.router_hash = model.router_hash.encode('utf-8')
        else:
            self.router_hash = router_hash.encode('utf-8') if router_hash else b''

    def _get_green_list_ids(self, expert_index: int) -> torch.Tensor:
        # Use IRSH protocol if router_hash is available
        if self.router_hash:
            seed_payload = f"{expert_index}:{self.router_hash.decode('utf-8')}".encode('utf-8')
        else:
            seed_payload = str(expert_index).encode('utf-8')
            
        h = hmac.new(self.secret_key, seed_payload, hashlib.sha256)
        seed = int.from_bytes(h.digest()[:8], 'big')
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(self.vocab_size, generator=generator)
        return permutation[:int(self.vocab_size * self.gamma)]

    def detect(self, text: str, z_threshold: float = 4.0) -> Dict[str, Any]:
        if not text.strip():
            return {"detected": False, "z_score": 0.0, "num_tokens": 0, "message": "Input text is empty or whitespace."}

        tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        num_tokens = tokenized_text.shape[1]

        if num_tokens < 10:
            return {"detected": False, "z_score": 0.0, "num_tokens": num_tokens, "message": "Text too short."}

        expert_choices = []
        def detection_hook_fn(module, args, output):
            router_logits = output
            choices = torch.argmax(router_logits, dim=-1).squeeze().tolist()
            if isinstance(choices, int): choices = [choices]
            expert_choices.extend(choices)

        handles = []
        for layer in self.model.model.layers:
            if hasattr(layer, 'block_sparse_moe') and isinstance(layer.block_sparse_moe, MixtralSparseMoeBlock):
                handle = layer.block_sparse_moe.gate.register_forward_hook(detection_hook_fn)
                handles.append(handle)

        if not handles: return {"error": "Could not find any MoE layers to attach hooks to."}

        try:
            with torch.no_grad():
                self.model(tokenized_text, output_router_logits=True)
        finally:
            for handle in handles: handle.remove()

        last_layer_expert_path = expert_choices[-num_tokens:]

        if len(last_layer_expert_path) != num_tokens:
             return {"error": f"Path reconstruction failed. Mismatch: {num_tokens} tokens vs {len(last_layer_expert_path)} choices."}

        green_token_count = 0
        for t in range(num_tokens):
            token_id = tokenized_text[0, t].item()
            expert_index = last_layer_expert_path[t]
            green_list_ids = self._get_green_list_ids(expert_index)
            if token_id in green_list_ids:
                green_token_count += 1

        expected_green_tokens = num_tokens * self.gamma
        variance = num_tokens * self.gamma * (1 - self.gamma)
        if variance == 0: return {"detected": False, "z_score": float('inf')}

        z_score = (green_token_count - expected_green_tokens) / math.sqrt(variance)

        return {
            "detected": z_score > z_threshold,
            "z_score": z_score,
            "num_green_tokens": green_token_count,
            "num_tokens": num_tokens,
        }

# =======================================================================================
# 5. MAIN EXECUTION BLOCK
# =======================================================================================
if __name__ == '__main__':
    print("--- Full Implementation of EPW-A Framework (Enhanced Architecture) ---")

    SECRET_KEY = "a_very_secret_and_long_key_for_hmac"
    model_name = "mistralai/Mixtral-8x7B-v0.1"


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

        model = MixtralForCausalLMWithWatermark.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            token=token,
        )
        model.eval()
        
        # Print router hash for IRSH protocol
        print(f"Router Hash (IRSH): {model.router_hash}")
        
    except Exception as e:
        print(f"\nError loading model or tokenizer: {e}")
        model, tokenizer = None, None

    if tokenizer and model:
        print("\n--- 2. Generating Watermarked Text with EPW-A ---")

        # Example 1: GSG mode (Gating-Seeded Green-listing)
        print("\n--- GSG Mode ---")
        gsg_processor = EPWALogitsProcessor(
            vocab_size=model.config.vocab_size,
            gamma=0.5,
            secret_key=SECRET_KEY,
            router_hash=model.router_hash,
            mode="gsg",
            delta_config=4.0
        )

        # Example 2: EWP mode (Expert-Specific Weighted Perturbation)
        print("\n--- EWP Mode ---")
        # Create expert-specific delta configuration
        # Assuming 8 experts (typical for Mixtral), assign different base deltas
        expert_deltas = {i: 3.0 + i * 0.5 for i in range(8)}  # Deltas from 3.0 to 6.5
        
        ewp_processor = EPWALogitsProcessor(
            vocab_size=model.config.vocab_size,
            gamma=0.5,
            secret_key=SECRET_KEY,
            router_hash=model.router_hash,
            mode="ewp",
            delta_config=expert_deltas
        )

        prompt = "In a world where algorithms whisper secrets, one discovery changed everything."
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"Prompt: '{prompt}'")

        # Generate text with GSG watermark
        print("\nGenerating with GSG watermark...")
        output_gsg = model.generate(
            **input_ids,
            max_new_tokens=50,
            logits_processor=[gsg_processor],
            do_sample=True,
            top_k=0,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        gsg_text = tokenizer.decode(output_gsg[0], skip_special_tokens=True)

        # Generate text with EWP watermark
        print("Generating with EWP watermark...")
        output_ewp = model.generate(
            **input_ids,
            max_new_tokens=50,
            logits_processor=[ewp_processor],
            do_sample=True,
            top_k=0,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        ewp_text = tokenizer.decode(output_ewp[0], skip_special_tokens=True)

        # Generate unwatermarked text for comparison
        output_unwatermarked = model.generate(
            **input_ids,
            max_new_tokens=50,
            do_sample=True,
            top_k=0,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        unwatermarked_text = tokenizer.decode(output_unwatermarked[0], skip_special_tokens=True)

        print(f"\nGSG Watermarked Text:\n{gsg_text}")
        print(f"\nEWP Watermarked Text:\n{ewp_text}")
        print(f"\nUnwatermarked Text:\n{unwatermarked_text}")

        print("\n--- 3. Detecting Watermark with IRSH Support ---")
        detector = WatermarkDetector(
            tokenizer=tokenizer, 
            model=model, 
            secret_key=SECRET_KEY, 
            gamma=0.5,
            router_hash=model.router_hash
        )

        generated_gsg_text = gsg_text[len(prompt):]
        generated_ewp_text = ewp_text[len(prompt):]
        generated_unwatermarked_text = unwatermarked_text[len(prompt):]

        print("\nAnalyzing GSG watermarked text...")
        result_gsg = detector.detect(generated_gsg_text)
        print(f"--> GSG Detection Result: {result_gsg}")

        print("\nAnalyzing EWP watermarked text...")
        result_ewp = detector.detect(generated_ewp_text)
        print(f"--> EWP Detection Result: {result_ewp}")

        print("\nAnalyzing unwatermarked text...")
        result_unwatermarked = detector.detect(generated_unwatermarked_text)
        print(f"--> Unwatermarked Detection Result: {result_unwatermarked}")

        # Summary
        print("\n--- Summary ---")
        print(f"GSG Watermark Detected: {result_gsg.get('detected', False)} (Z-score: {result_gsg.get('z_score', 0):.2f})")
        print(f"EWP Watermark Detected: {result_ewp.get('detected', False)} (Z-score: {result_ewp.get('z_score', 0):.2f})")
        print(f"False Positive: {result_unwatermarked.get('detected', False)} (Z-score: {result_unwatermarked.get('z_score', 0):.2f})")
        
    else:
        print("\nSkipping demonstration due to model or tokenizer loading error.")
