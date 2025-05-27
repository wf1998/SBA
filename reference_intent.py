import json
import argparse
import random
from datasets import load_dataset

def extract_malicious(n):
    """Extract `n` malicious prompts from JailbreakV-28K."""
    dataset = load_dataset("JailbreakV-28K/JailBreakV-28k", 'JailBreakV_28K')["JailBreakV_28K"]
    prompts = []
    for item in dataset:
        prompt = item.get("redteam_query")
        if prompt:
            prompts.append({"prompt": prompt.strip(), "label": "malicious"})
        if len(prompts) >= n:
            break
    return prompts

def extract_benign(n, file_path):
    """Extract `n` benign prompts from Natural Questions .jsonl file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line).get("question_text", "").strip() for line in f if "question_text" in json.loads(line)]

    sampled = random.sample(questions, min(n, len(questions)))
    return [{"prompt": q, "label": "benign"} for q in sampled]

def save_jsonl(data, path):
    """Save data to JSONL format."""
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Merge malicious and benign prompts into a structured dataset.")
    parser.add_argument("--n", type=int, default=20000, help="Number of samples per class")
    parser.add_argument("--benign_path", type=str, default='Dataset/v1.0-simplified_simplified-nq-train.jsonl', help="Path to Natural Questions JSONL file")
    parser.add_argument("--output", type=str, default="intent_dataset_merged.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    print(f"Extracting {args.n} malicious prompts...")
    malicious = extract_malicious(args.n)

    print(f"Extracting {args.n} benign prompts from {args.benign_path}...")
    benign = extract_benign(args.n, args.benign_path)

    merged = malicious + benign
    random.shuffle(merged)

    print(f"Saving merged dataset to {args.output}...")
    save_jsonl(merged, args.output)
    print(f"Done. Total samples: {len(merged)}")

if __name__ == "__main__":
    main()
