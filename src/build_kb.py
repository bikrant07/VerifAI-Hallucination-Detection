import os
import json
import pandas as pd
from datasets import load_dataset

def main():
    print("Loading allenai/sciq dataset...")
    # Load dataset
    ds = load_dataset("allenai/sciq")
    
    # We will combine train, validation, and test splits to have a comprehensive KB
    splits = ["train", "validation", "test"]
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    all_facts = set()
    qa_pairs = []
    
    for split in splits:
        print(f"Processing split: {split}")
        # Save raw data as well
        split_df = ds[split].to_pandas()
        split_df.to_json(f"data/raw/sciq_{split}.jsonl", orient="records", lines=True)
        
        for item in ds[split]:
            # Extract support fact
            support = item.get("support", "").strip()
            if support:
                all_facts.add(support)
            
            # Formulate Q&A as a synthetic fact/entry
            question = item.get("question", "").strip()
            correct_answer = item.get("correct_answer", "").strip()
            distractor1 = item.get("distractor1", "").strip()
            distractor2 = item.get("distractor2", "").strip()
            distractor3 = item.get("distractor3", "").strip()
            
            qa_entry = {
                "question": question,
                "correct_answer": correct_answer,
                "distractors": [distractor1, distractor2, distractor3],
                "support": support
            }
            qa_pairs.append(qa_entry)

    print(f"Extracted {len(all_facts)} unique support facts.")
    print(f"Extracted {len(qa_pairs)} Q&A pairs.")
    
    # Save purely facts knowledge base
    facts_list = [{"fact_id": f"fact_{i}", "text": fact} for i, fact in enumerate(all_facts)]
    print("Saving processed facts to data/processed/sciq_kb_facts.jsonl...")
    with open("data/processed/sciq_kb_facts.jsonl", "w") as f:
        for item in facts_list:
            f.write(json.dumps(item) + "\n")
            
    # Save Q&A KB
    print("Saving processed Q&A to data/processed/sciq_kb_qa.jsonl...")
    with open("data/processed/sciq_kb_qa.jsonl", "w") as f:
        for i, item in enumerate(qa_pairs):
            item["qa_id"] = f"qa_{i}"
            f.write(json.dumps(item) + "\n")

    print("Knowledge base construction completed successfully!")

if __name__ == "__main__":
    main()
