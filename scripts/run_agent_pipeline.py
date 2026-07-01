import os
import json
import torch
import sys
from unsloth import FastLanguageModel

# Add src to path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sentinel_inference import get_timesfm_model
import numpy as np

def run_agentic_pipeline(model_id="gemma4_unified", lora_weights_path="outputs/dikd-gemma4-12b/lora_adapter"):
    print("=== Loading TimesFM Sentinel Model ===")
    timesfm_model = get_timesfm_model()

    print("\n=== Loading Fine-Tuned Clinical LLM (Gemma 4) ===")
    max_seq_length = 2048
    dtype = None # Auto detection
    load_in_4bit = True # Use 4bit quantization to save memory

    # Load the base model and merged adapters
    # Make sure this points to where your model gets saved after training/merging
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_weights_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    def ask_timesfm(patient_data):
        """
        Tool callable by the LLM. 
        patient_data should be a dict with context arrays for: scr, map, vanco_trough, zosyn_active
        """
        print("\n[Tool Call] The LLM is querying TimesFM for a 48-hour SCr forecast...")
        
        # Prepare inputs exactly as the LoRA adapters were trained!
        context = np.array(patient_data['scr'][:3], dtype=np.float32)
        covs = np.zeros((3, 3), dtype=np.float32)
        covs[:, 0] = patient_data['map'][:3]
        covs[:, 1] = patient_data['vanco_trough'][:3]
        covs[:, 2] = [1.0 if x else 0.0 for x in patient_data['zosyn_active'][:3]]
        
        padded_input = np.zeros(63, dtype=np.float32)
        padded_input[:len(context)] = context
        padded_input[3:6] = covs[:, 0]
        padded_input[6:9] = covs[:, 1]
        padded_input[9:12] = covs[:, 2]
        
        # Access the raw PyTorch module with LoRA attached
        pytorch_module = getattr(timesfm_model, 'model', timesfm_model)
        device = next(pytorch_module.parameters()).device
        
        inputs = torch.tensor(padded_input, device=device).unsqueeze(0).unsqueeze(0)
        masks = torch.ones(1, 1, 1, dtype=torch.float32, device=device)
        
        # Manual Forward Pass
        out = pytorch_module(inputs, masks)
        preds = out[0][2].squeeze().flatten()
        day_4_pred = preds[0].item()
        day_5_pred = preds[1].item()
        
        forecast_result = f"TimesFM Forecast: SCr on Day 4: {day_4_pred:.2f} mg/dL, Day 5: {day_5_pred:.2f} mg/dL."
        print(f"[Tool Response] {forecast_result}")
        return forecast_result

    # Let's simulate the LLM reading a chart at Day 3
    print("\n=== Simulating Virtual Pharmacist Agent ===")
    
    # Fake a high risk patient on day 3
    high_risk_patient = {
        'scr': [1.0, 1.1, 1.3, 0.0, 0.0], # Day 4, 5 unknown to context
        'map': [80, 75, 70, 70, 70],
        'vanco_trough': [15.0, 18.0, 22.0, 22.0, 22.0],
        'zosyn_active': [True, True, True, True, True]
    }
    
    patient_prompt = f"""
    You are an AI Clinical Pharmacist reviewing a patient's chart at the end of Day 3.
    Patient is actively receiving Vancomycin and Piperacillin-Tazobactam (Zosyn).
    Current SCr trend: Day 1 (1.0), Day 2 (1.1), Day 3 (1.3).
    Vancomycin Trough is rising, currently at 22.0 ug/mL.
    
    Please assess the risk of nephrotoxicity.
    """
    
    # System prompt instructing the LLM to use the TimesFM tool
    messages = [
        {"role": "system", "content": "You are a clinical pharmacist. You have access to a TimesFM forecasting tool. Base your assessment on the patient's Day 1-3 data, but ALWAYS query the TimesFM tool to predict Days 4-5 before making a final clinical recommendation."},
        {"role": "user", "content": patient_prompt}
    ]
    
    # 1. The LLM decides it needs the forecast based on instructions
    # (In a true agentic framework like LangChain, this would be an automatic tool call loop)
    # Here, we simulate the LLM making the tool call because it sees the prompt
    timesfm_forecast_text = ask_timesfm(high_risk_patient)
    
    # 2. We inject the tool response back into the LLM context
    messages.append({"role": "assistant", "content": "I need to run a TimesFM forecast based on this Day 1-3 data."})
    messages.append({"role": "user", "content": f"Tool Response: {timesfm_forecast_text}"})
    
    print("\n[LLM] Generating Final Clinical Note...")
    # 3. The LLM generates the final note using the forecast
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, 
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 256, use_cache = True)
    response = tokenizer.batch_decode(outputs)[0]
    
    # Extract just the assistant's final response
    final_note = response.split("<start_of_turn>model")[-1].replace("<end_of_turn>", "").strip()
    
    print("\n" + "="*50)
    print("FINAL CLINICAL NOTE:")
    print("="*50)
    print(final_note)

if __name__ == "__main__":
    # Assumes the merged model is at output/gemma4-12b-aki
    run_agentic_pipeline()
