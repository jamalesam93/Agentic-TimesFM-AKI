import json
import re

def calculate_metrics():
    input_file = "reports/ablation_eval_predictions.jsonl"
    
    total = 0
    parsed = 0
    correct = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            data = json.loads(line)
            
            gt = data.get("ground_truth")
            pred = data.get("predicted")
            
            if pred != "PARSE_ERROR":
                parsed += 1
                if pred == gt:
                    correct += 1
                    
    print(f"Total: {total}")
    print(f"Parsed: {parsed} ({parsed/total*100:.1f}%)")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Accuracy (on parsed): {correct/parsed*100:.1f}%")

if __name__ == "__main__":
    calculate_metrics()
