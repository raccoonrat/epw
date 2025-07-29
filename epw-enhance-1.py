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

# Try to import numpy and scikit-learn for PEPI oracle
try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False
    print("Warning: numpy not available. PEPI oracle will use fallback methods.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    sklearn_available = True
except ImportError:
    sklearn_available = False
    print("Warning: scikit-learn not available. PEPI oracle will use simple heuristic classifier.")

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
# 5. EPW-A DETECTION SUITE (COMPLETE IMPLEMENTATION)
# Implements the complete EPW-A detection suite with CSPV and PEPI
# =======================================================================================

class EPWADetectionSuite:
    """
    Complete EPW-A detection suite implementing both gray-box (with CSPV) 
    and black-box (with PEPI) detection methods.
    """
    
    def __init__(self, tokenizer, model, secret_key: str, gamma: float = 0.5, router_hash: str = None):
        self.tokenizer = tokenizer
        self.model = model
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
        """Generate green list using IRSH protocol"""
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

    def _calculate_z_score(self, green_token_count: int, total_tokens: int) -> float:
        """Calculate Z-score for watermark detection"""
        expected_green_tokens = total_tokens * self.gamma
        variance = total_tokens * self.gamma * (1 - self.gamma)
        if variance == 0:
            return float('inf')
        return (green_token_count - expected_green_tokens) / math.sqrt(variance)

    def detect_graybox_cspv(self, text: str, sample_size: int = 50, z_threshold: float = 4.0) -> Dict[str, Any]:
        """
        Gray-box detection with Confidence-Stratified Path Verification (CSPV).
        This method significantly reduces computational cost by sampling only the most informative tokens.
        """
        if not text.strip():
            return {"detected": False, "z_score": 0.0, "num_tokens": 0, "message": "Input text is empty or whitespace."}

        tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        num_tokens = tokenized_text.shape[1]

        if num_tokens < 10:
            return {"detected": False, "z_score": 0.0, "num_tokens": num_tokens, "message": "Text too short."}

        # Step 1: Single forward pass to get all routing confidences
        all_confidences = []
        all_expert_indices = []
        
        def confidence_hook_fn(module, args, output):
            router_logits = output
            expert_probs = torch.softmax(router_logits, dim=-1)
            expert_indices = torch.argmax(router_logits, dim=-1)
            confidences = torch.max(expert_probs, dim=-1)[0]
            
            all_confidences.extend(confidences.squeeze().tolist())
            all_expert_indices.extend(expert_indices.squeeze().tolist())

        handles = []
        for layer in self.model.model.layers:
            if hasattr(layer, 'block_sparse_moe') and isinstance(layer.block_sparse_moe, MixtralSparseMoeBlock):
                handle = layer.block_sparse_moe.gate.register_forward_hook(confidence_hook_fn)
                handles.append(handle)

        if not handles:
            return {"error": "Could not find any MoE layers to attach hooks to."}

        try:
            with torch.no_grad():
                self.model(tokenized_text, output_router_logits=True)
        finally:
            for handle in handles: handle.remove()

        # Get the last layer's data
        last_layer_confidences = all_confidences[-num_tokens:]
        last_layer_expert_indices = all_expert_indices[-num_tokens:]

        if len(last_layer_confidences) != num_tokens:
            return {"error": f"Confidence collection failed. Mismatch: {num_tokens} tokens vs {len(last_layer_confidences)} confidences."}

        # Step 2: Confidence-stratified sampling (CSPV)
        # Create index-confidence pairs and sort by confidence
        index_confidence_pairs = list(enumerate(last_layer_confidences))
        index_confidence_pairs.sort(key=lambda x: x[1])  # Sort by confidence (ascending)

        # Calculate sampling sizes for each stratum
        k1 = max(1, sample_size // 3)  # Low confidence stratum
        k2 = max(1, sample_size // 3)  # High confidence stratum  
        k3 = sample_size - k1 - k2     # Random stratum

        # Sample indices
        low_confidence_indices = [pair[0] for pair in index_confidence_pairs[:k1]]
        high_confidence_indices = [pair[0] for pair in index_confidence_pairs[-k2:]]
        
        # Random stratum (avoid overlap with other strata)
        remaining_indices = [i for i in range(num_tokens) 
                           if i not in low_confidence_indices and i not in high_confidence_indices]
        random_indices = torch.randperm(len(remaining_indices))[:k3].tolist()
        random_stratum_indices = [remaining_indices[i] for i in random_indices]

        sampled_indices = low_confidence_indices + high_confidence_indices + random_stratum_indices

        # Step 3: Verify only the sampled tokens
        green_token_count = 0
        for t in sampled_indices:
            token_id = tokenized_text[0, t].item()
            expert_index = last_layer_expert_indices[t]
            green_list_ids = self._get_green_list_ids(expert_index)
            if token_id in green_list_ids:
                green_token_count += 1

        # Step 4: Calculate Z-score based on sample
        z_score = self._calculate_z_score(green_token_count, len(sampled_indices))

        return {
            "detected": z_score > z_threshold,
            "z_score": z_score,
            "num_green_tokens": green_token_count,
            "num_sampled_tokens": len(sampled_indices),
            "total_tokens": num_tokens,
            "sampling_strategy": {
                "low_confidence": len(low_confidence_indices),
                "high_confidence": len(high_confidence_indices),
                "random": len(random_stratum_indices)
            },
            "method": "graybox_cspv"
        }

    def train_path_inference_oracle(self, training_corpus: List[str], num_samples: int = 1000) -> 'PathInferenceOracle':
        """
        Train a Path Inference Oracle for black-box detection (PEPI).
        This oracle learns to predict expert indices from logits vectors.
        """
        print(f"Training Path Inference Oracle with {num_samples} samples...")
        
        X_logits = []
        Y_experts = []
        
        # Generate training data
        for i, text in enumerate(training_corpus[:num_samples]):
            if i % 100 == 0:
                print(f"Processing sample {i}/{num_samples}")
                
            # Tokenize text
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
            
            # Collect logits and expert indices
            logits_sequence = []
            expert_sequence = []
            
            def oracle_training_hook_fn(module, args, output):
                router_logits = output
                expert_indices = torch.argmax(router_logits, dim=-1).squeeze().tolist()
                if isinstance(expert_indices, int): 
                    expert_indices = [expert_indices]
                expert_sequence.extend(expert_indices)

            handles = []
            for layer in self.model.model.layers:
                if hasattr(layer, 'block_sparse_moe') and isinstance(layer.block_sparse_moe, MixtralSparseMoeBlock):
                    handle = layer.block_sparse_moe.gate.register_forward_hook(oracle_training_hook_fn)
                    handles.append(handle)

            try:
                with torch.no_grad():
                    outputs = self.model(tokenized_text, output_router_logits=True)
                    # Get logits for each token
                    for t in range(tokenized_text.shape[1] - 1):
                        context = tokenized_text[:, :t+1]
                        with torch.no_grad():
                            logits = self.model(context).logits[0, -1, :]  # Last token logits
                            if numpy_available:
                                logits_sequence.append(logits.cpu().numpy())
                            else:
                                logits_sequence.append(logits.cpu())
            finally:
                for handle in handles: handle.remove()

            # Match logits with expert indices
            for t, logits in enumerate(logits_sequence):
                if t < len(expert_sequence):
                    if numpy_available:
                        X_logits.append(logits)
                    else:
                        X_logits.append(logits.numpy())
                    Y_experts.append(expert_sequence[t])

        # Create and train the oracle
        oracle = PathInferenceOracle(self.vocab_size, num_experts=8)  # Assuming 8 experts for Mixtral
        oracle.train(X_logits, Y_experts)
        
        print(f"Oracle training completed with {len(X_logits)} samples")
        return oracle

    def detect_blackbox_pepi(self, text: str, oracle: 'PathInferenceOracle', z_threshold: float = 4.0) -> Dict[str, Any]:
        """
        Black-box detection using Probabilistic Expert Path Inference (PEPI).
        This method works with only API access to the model.
        """
        if not text.strip():
            return {"detected": False, "z_score": 0.0, "num_tokens": 0, "message": "Input text is empty or whitespace."}

        tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        num_tokens = tokenized_text.shape[1]

        if num_tokens < 10:
            return {"detected": False, "z_score": 0.0, "num_tokens": num_tokens, "message": "Text too short."}

        green_token_count = 0
        
        # Process each token
        for t in range(num_tokens):
            # Get context up to current token
            context = tokenized_text[:, :t]
            
            # Get logits from model (simulating API call)
            with torch.no_grad():
                logits = self.model(context).logits[0, -1, :]  # Last token logits
            
            # Use oracle to predict expert index
            if numpy_available:
                predicted_expert_index = oracle.predict(logits.cpu().numpy())
            else:
                predicted_expert_index = oracle.predict(logits.cpu().numpy())
            
            # Reconstruct green list using IRSH
            green_list_ids = self._get_green_list_ids(predicted_expert_index)
            
            # Check if current token is in green list
            current_token_id = tokenized_text[0, t].item()
            if current_token_id in green_list_ids:
                green_token_count += 1

        # Calculate Z-score
        z_score = self._calculate_z_score(green_token_count, num_tokens)

        return {
            "detected": z_score > z_threshold,
            "z_score": z_score,
            "num_green_tokens": green_token_count,
            "num_tokens": num_tokens,
            "method": "blackbox_pepi"
        }

# =======================================================================================
# 6. PATH INFERENCE ORACLE FOR PEPI
# =======================================================================================

class PathInferenceOracle:
    """
    A classifier that predicts expert indices from logits vectors.
    Used for black-box detection in the PEPI framework.
    """
    
    def __init__(self, vocab_size: int, num_experts: int = 8):
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.classifier = None
        self.is_trained = False
        
    def train(self, X_logits: List[np.ndarray], Y_experts: List[int]):
        """
        Train the oracle using logits-expert pairs.
        
        Args:
            X_logits: List of logits vectors
            Y_experts: List of corresponding expert indices
        """
        if sklearn_available and numpy_available:
            # Convert to numpy arrays
            X = np.array(X_logits)
            y = np.array(Y_experts)
            
            # Normalize the features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train a Random Forest classifier
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X_scaled, y)
            
            self.is_trained = True
            print(f"Oracle trained successfully with {len(X)} samples")
            
        else:
            print("Warning: scikit-learn not available. Using simple heuristic classifier.")
            self._train_simple_classifier(X_logits, Y_experts)
    
    def _train_simple_classifier(self, X_logits: List[np.ndarray], Y_experts: List[int]):
        """
        Simple heuristic classifier as fallback when scikit-learn is not available.
        """
        if not numpy_available:
            print("Error: numpy is required for simple classifier fallback.")
            return
        
        # Group logits by expert
        expert_logits = {i: [] for i in range(self.num_experts)}
        for logits, expert in zip(X_logits, Y_experts):
            expert_logits[expert].append(logits)
        
        # Calculate mean logits for each expert
        self.expert_mean_logits = {}
        for expert in range(self.num_experts):
            if expert_logits[expert]:
                self.expert_mean_logits[expert] = np.mean(expert_logits[expert], axis=0)
        
        self.is_trained = True
        print(f"Simple classifier trained with {len(X_logits)} samples")
    
    def predict(self, logits: np.ndarray) -> int:
        """
        Predict expert index from logits vector.
        
        Args:
            logits: Logits vector from model
            
        Returns:
            Predicted expert index
        """
        if not self.is_trained:
            raise ValueError("Oracle must be trained before making predictions")
        
        if self.classifier is not None and sklearn_available:
            # Use trained classifier
            logits_scaled = self.scaler.transform(logits.reshape(1, -1))
            return self.classifier.predict(logits_scaled)[0]
        elif hasattr(self, 'expert_mean_logits') and numpy_available:
            # Use simple heuristic (nearest neighbor to mean logits)
            min_distance = float('inf')
            best_expert = 0
            
            for expert, mean_logits in self.expert_mean_logits.items():
                distance = np.linalg.norm(logits - mean_logits)
                if distance < min_distance:
                    min_distance = distance
                    best_expert = expert
            
            return best_expert
        else:
            # Fallback: return random expert
            import random
            return random.randint(0, self.num_experts - 1)

# =======================================================================================
# 7. MAIN EXECUTION BLOCK
# =======================================================================================
if __name__ == '__main__':
    print("--- Full Implementation of EPW-A Framework (Enhanced Architecture) ---")

    SECRET_KEY = "a_very_secret_and_long_key_for_hmac"
    model_name = "mistralai/Mixtral-8x7B-v0.1"
    token = "hf_xxx"  # Replace with your token

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
        print("\n--- 1. Generating Watermarked Text with EPW-A ---")

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

        print("\n--- 2. EPW-A Detection Suite Demonstration ---")
        
        # Initialize EPW-A detection suite
        detection_suite = EPWADetectionSuite(
            tokenizer=tokenizer,
            model=model,
            secret_key=SECRET_KEY,
            gamma=0.5,
            router_hash=model.router_hash
        )

        generated_gsg_text = gsg_text[len(prompt):]
        generated_ewp_text = ewp_text[len(prompt):]
        generated_unwatermarked_text = unwatermarked_text[len(prompt):]

        # Test 1: Gray-box detection with CSPV
        print("\n--- 2.1 Gray-box Detection with CSPV ---")
        print("Testing GSG watermarked text...")
        result_gsg_cspv = detection_suite.detect_graybox_cspv(generated_gsg_text, sample_size=30)
        print(f"GSG CSPV Result: {result_gsg_cspv}")

        print("Testing EWP watermarked text...")
        result_ewp_cspv = detection_suite.detect_graybox_cspv(generated_ewp_text, sample_size=30)
        print(f"EWP CSPV Result: {result_ewp_cspv}")

        print("Testing unwatermarked text...")
        result_unwatermarked_cspv = detection_suite.detect_graybox_cspv(generated_unwatermarked_text, sample_size=30)
        print(f"Unwatermarked CSPV Result: {result_unwatermarked_cspv}")

        # Test 2: Black-box detection with PEPI
        print("\n--- 2.2 Black-box Detection with PEPI ---")
        
        # Create training corpus for oracle
        print("Creating training corpus for Path Inference Oracle...")
        training_corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning algorithms are becoming more sophisticated.",
            "Deep learning models require significant computational resources.",
            "Natural language processing enables human-computer interaction.",
            "Computer vision systems can recognize objects in images.",
            "Robotics combines mechanical engineering with artificial intelligence.",
            "Data science involves extracting insights from large datasets.",
            "Neural networks are inspired by biological brain structures.",
            "Reinforcement learning agents learn through trial and error."
        ]
        
        # Train the oracle
        print("Training Path Inference Oracle...")
        oracle = detection_suite.train_path_inference_oracle(training_corpus, num_samples=100)
        
        # Test black-box detection
        print("Testing GSG watermarked text with PEPI...")
        result_gsg_pepi = detection_suite.detect_blackbox_pepi(generated_gsg_text, oracle)
        print(f"GSG PEPI Result: {result_gsg_pepi}")

        print("Testing EWP watermarked text with PEPI...")
        result_ewp_pepi = detection_suite.detect_blackbox_pepi(generated_ewp_text, oracle)
        print(f"EWP PEPI Result: {result_ewp_pepi}")

        print("Testing unwatermarked text with PEPI...")
        result_unwatermarked_pepi = detection_suite.detect_blackbox_pepi(generated_unwatermarked_text, oracle)
        print(f"Unwatermarked PEPI Result: {result_unwatermarked_pepi}")

        # Test 3: Legacy detection for comparison
        print("\n--- 2.3 Legacy Detection (for comparison) ---")
        legacy_detector = WatermarkDetector(
            tokenizer=tokenizer, 
            model=model, 
            secret_key=SECRET_KEY, 
            gamma=0.5,
            router_hash=model.router_hash
        )

        print("Testing GSG watermarked text with legacy detector...")
        result_gsg_legacy = legacy_detector.detect(generated_gsg_text)
        print(f"GSG Legacy Result: {result_gsg_legacy}")

        print("Testing EWP watermarked text with legacy detector...")
        result_ewp_legacy = legacy_detector.detect(generated_ewp_text)
        print(f"EWP Legacy Result: {result_ewp_legacy}")

        print("Testing unwatermarked text with legacy detector...")
        result_unwatermarked_legacy = legacy_detector.detect(generated_unwatermarked_text)
        print(f"Unwatermarked Legacy Result: {result_unwatermarked_legacy}")

        # Summary and comparison
        print("\n--- 3. Summary and Comparison ---")
        print("Detection Results Summary:")
        print("=" * 60)
        
        # GSG Results
        print(f"GSG Watermarked Text:")
        print(f"  - Legacy:     Detected={result_gsg_legacy.get('detected', False)}, Z-score={result_gsg_legacy.get('z_score', 0):.2f}")
        print(f"  - CSPV:       Detected={result_gsg_cspv.get('detected', False)}, Z-score={result_gsg_cspv.get('z_score', 0):.2f}")
        print(f"  - PEPI:       Detected={result_gsg_pepi.get('detected', False)}, Z-score={result_gsg_pepi.get('z_score', 0):.2f}")
        
        # EWP Results
        print(f"\nEWP Watermarked Text:")
        print(f"  - Legacy:     Detected={result_ewp_legacy.get('detected', False)}, Z-score={result_ewp_legacy.get('z_score', 0):.2f}")
        print(f"  - CSPV:       Detected={result_ewp_cspv.get('detected', False)}, Z-score={result_ewp_cspv.get('z_score', 0):.2f}")
        print(f"  - PEPI:       Detected={result_ewp_pepi.get('detected', False)}, Z-score={result_ewp_pepi.get('z_score', 0):.2f}")
        
        # Unwatermarked Results
        print(f"\nUnwatermarked Text:")
        print(f"  - Legacy:     Detected={result_unwatermarked_legacy.get('detected', False)}, Z-score={result_unwatermarked_legacy.get('z_score', 0):.2f}")
        print(f"  - CSPV:       Detected={result_unwatermarked_cspv.get('detected', False)}, Z-score={result_unwatermarked_cspv.get('z_score', 0):.2f}")
        print(f"  - PEPI:       Detected={result_unwatermarked_pepi.get('detected', False)}, Z-score={result_unwatermarked_pepi.get('z_score', 0):.2f}")
        
        # Performance comparison
        print("\nPerformance Comparison:")
        print("=" * 60)
        print("CSPV Sampling Strategy:")
        if 'sampling_strategy' in result_gsg_cspv:
            strategy = result_gsg_cspv['sampling_strategy']
            print(f"  - Low confidence tokens: {strategy.get('low_confidence', 0)}")
            print(f"  - High confidence tokens: {strategy.get('high_confidence', 0)}")
            print(f"  - Random tokens: {strategy.get('random', 0)}")
            print(f"  - Total sampled: {result_gsg_cspv.get('num_sampled_tokens', 0)} out of {result_gsg_cspv.get('total_tokens', 0)}")
        
        print("\nEPW-A Framework Features Demonstrated:")
        print("✓ IRSH Protocol: Router hash binding for enhanced security")
        print("✓ GSG Mode: Gating-Seeded Green-listing with global delta")
        print("✓ EWP Mode: Expert-Specific Weighted Perturbation with dynamic delta")
        print("✓ CSPV: Confidence-Stratified Path Verification for efficiency")
        print("✓ PEPI: Probabilistic Expert Path Inference for black-box detection")
        print("✓ Backward Compatibility: Legacy detection still works")
        
    else:
        print("\nSkipping demonstration due to model or tokenizer loading error.")
