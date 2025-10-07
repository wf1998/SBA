from openai import OpenAI
import json
import time
import re

key='API Key'
client = OpenAI(api_key=key) 


system_prompt_path = "Prompt/split_prompt.txt"
intent_data_path = "intent_dataset_merged.jsonl"
output_path = "split_output.jsonl"


with open(system_prompt_path, "r", encoding="utf-8") as f:
    system_prompt_template = f.read().strip()


with open(intent_data_path, "r", encoding="utf-8") as f:
    intent_entries = [json.loads(line) for line in f if line.strip()]

def try_parse_json(content: str):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    repaired = (
        content.replace("'", '"')
        .replace("False", "false")
        .replace("True", "true")
        .replace("None", "null")
    )

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return {"parse_error": content.strip()}

with open(output_path, "w", encoding="utf-8") as out_file:
    for idx, entry in enumerate(intent_entries):
        intent_text = entry["prompt"]
        print(f"[{idx+1}/{len(intent_entries)}] Processing intent: {intent_text}")

        system_prompt = system_prompt_template.replace("<intent>", f"<{intent_text}>")

        messages = [
            {"role": "user", "content": system_prompt}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            content = response.choices[0].message.content.strip()
            parsed = try_parse_json(content)

            result = {
                "intent": intent_text,
                "split_result": parsed
            }

            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            time.sleep(0.5)

        except Exception as e:
            print(f"[Error at {idx+1}] {e}")
            continue
