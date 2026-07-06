import json
import re

def create_ablation_dataset():
    input_file = "data/real_world/paper_eval_holdout.jsonl"
    output_file = "data/real_world/ablation_eval_holdout_3day.jsonl"
    
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            user_content = data["messages"][1]["content"]
            
            # Remove Day 4 and Day 5 lines
            user_content = re.sub(r"\[Day 4\].*\n?", "", user_content)
            user_content = re.sub(r"\[Day 5\].*\n?", "", user_content)
            
            data["messages"][1]["content"] = user_content.strip()
            f_out.write(json.dumps(data) + "\n")
            
if __name__ == "__main__":
    create_ablation_dataset()
    print("Created ablation_eval_holdout_3day.jsonl")
