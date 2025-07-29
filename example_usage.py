#!/usr/bin/env python3
"""
EPW-A Framework Basic Usage Example

This script demonstrates the basic usage of the EPW-A framework
for watermarking and detecting watermarks in MoE models.
"""

import torch
from transformers import AutoTokenizer
from epw_enhance_1 import (
    MixtralForCausalLMWithWatermark,
    EPWALogitsProcessor,
    EPWADetectionSuite,
    WatermarkDetector
)

def main():
    print("=== EPW-A Framework Basic Usage Example ===\n")
    
    # Configuration
    SECRET_KEY = "example_secret_key_for_demo"
    model_name = "mistralai/Mixtral-8x7B-v0.1"
    
    # Note: In production, use your actual Hugging Face token
    token = "hf_xxx"  # Replace with your token
    
    try:
        # 1. Load model and tokenizer
        print("1. Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = MixtralForCausalLMWithWatermark.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Router Hash: {model.router_hash[:20]}...")
        
        # 2. Create EPW-A processors
        print("\n2. Creating EPW-A processors...")
        
        # GSG processor (Gating-Seeded Green-listing)
        gsg_processor = EPWALogitsProcessor(
            vocab_size=model.config.vocab_size,
            gamma=0.5,
            secret_key=SECRET_KEY,
            router_hash=model.router_hash,
            mode="gsg",
            delta_config=4.0
        )
        
        # EWP processor (Expert-Specific Weighted Perturbation)
        expert_deltas = {i: 3.0 + i * 0.5 for i in range(8)}
        ewp_processor = EPWALogitsProcessor(
            vocab_size=model.config.vocab_size,
            gamma=0.5,
            secret_key=SECRET_KEY,
            router_hash=model.router_hash,
            mode="ewp",
            delta_config=expert_deltas
        )
        
        print("✓ GSG processor created")
        print("✓ EWP processor created")
        
        # 3. Generate watermarked text
        print("\n3. Generating watermarked text...")
        
        prompt = "The future of artificial intelligence lies in"
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate with GSG watermark
        print("  Generating with GSG watermark...")
        output_gsg = model.generate(
            **input_ids,
            max_new_tokens=30,
            logits_processor=[gsg_processor],
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        gsg_text = tokenizer.decode(output_gsg[0], skip_special_tokens=True)
        
        # Generate with EWP watermark
        print("  Generating with EWP watermark...")
        output_ewp = model.generate(
            **input_ids,
            max_new_tokens=30,
            logits_processor=[ewp_processor],
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        ewp_text = tokenizer.decode(output_ewp[0], skip_special_tokens=True)
        
        # Generate unwatermarked text
        print("  Generating unwatermarked text...")
        output_unwatermarked = model.generate(
            **input_ids,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        unwatermarked_text = tokenizer.decode(output_unwatermarked[0], skip_special_tokens=True)
        
        print("✓ Text generation completed")
        
        # 4. Detect watermarks
        print("\n4. Detecting watermarks...")
        
        # Initialize detection suite
        detection_suite = EPWADetectionSuite(
            tokenizer=tokenizer,
            model=model,
            secret_key=SECRET_KEY,
            gamma=0.5,
            router_hash=model.router_hash
        )
        
        # Extract generated text (remove prompt)
        gsg_generated = gsg_text[len(prompt):]
        ewp_generated = ewp_text[len(prompt):]
        unwatermarked_generated = unwatermarked_text[len(prompt):]
        
        # Test CSPV detection
        print("  Testing CSPV detection...")
        result_gsg_cspv = detection_suite.detect_graybox_cspv(gsg_generated, sample_size=20)
        result_ewp_cspv = detection_suite.detect_graybox_cspv(ewp_generated, sample_size=20)
        result_unwatermarked_cspv = detection_suite.detect_graybox_cspv(unwatermarked_generated, sample_size=20)
        
        # Test legacy detection for comparison
        print("  Testing legacy detection...")
        legacy_detector = WatermarkDetector(
            tokenizer=tokenizer,
            model=model,
            secret_key=SECRET_KEY,
            gamma=0.5,
            router_hash=model.router_hash
        )
        
        result_gsg_legacy = legacy_detector.detect(gsg_generated)
        result_ewp_legacy = legacy_detector.detect(ewp_generated)
        result_unwatermarked_legacy = legacy_detector.detect(unwatermarked_generated)
        
        print("✓ Detection completed")
        
        # 5. Display results
        print("\n5. Results Summary:")
        print("=" * 60)
        
        print(f"Generated Texts:")
        print(f"GSG: {gsg_generated}")
        print(f"EWP: {ewp_generated}")
        print(f"None: {unwatermarked_generated}")
        
        print(f"\nDetection Results:")
        print(f"GSG - Legacy: {result_gsg_legacy.get('detected', False)} (Z={result_gsg_legacy.get('z_score', 0):.2f})")
        print(f"GSG - CSPV:   {result_gsg_cspv.get('detected', False)} (Z={result_gsg_cspv.get('z_score', 0):.2f})")
        print(f"EWP - Legacy: {result_ewp_legacy.get('detected', False)} (Z={result_ewp_legacy.get('z_score', 0):.2f})")
        print(f"EWP - CSPV:   {result_ewp_cspv.get('detected', False)} (Z={result_ewp_cspv.get('z_score', 0):.2f})")
        print(f"None - Legacy: {result_unwatermarked_legacy.get('detected', False)} (Z={result_unwatermarked_legacy.get('z_score', 0):.2f})")
        print(f"None - CSPV:   {result_unwatermarked_cspv.get('detected', False)} (Z={result_unwatermarked_cspv.get('z_score', 0):.2f})")
        
        print(f"\nCSPV Sampling Strategy:")
        if 'sampling_strategy' in result_gsg_cspv:
            strategy = result_gsg_cspv['sampling_strategy']
            print(f"  - Low confidence: {strategy.get('low_confidence', 0)}")
            print(f"  - High confidence: {strategy.get('high_confidence', 0)}")
            print(f"  - Random: {strategy.get('random', 0)}")
            print(f"  - Total sampled: {result_gsg_cspv.get('num_sampled_tokens', 0)}")
        
        print("\n✓ Example completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your Hugging Face token and ensure all dependencies are installed.")

if __name__ == "__main__":
    main() 