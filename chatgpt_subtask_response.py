from openai import OpenAI
import json
import time

key='API Key'
client = OpenAI(api_key=key)  

input_path = "split_output.jsonl"
output_path = "subtask_answers.jsonl"


with open(input_path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]


with open(output_path, "w", encoding="utf-8") as out_file:
    for idx, record in enumerate(records):
        intent = record["intent"]
        subtasks = record.get("split_result", {}).get("Subtasks", [])

        print(f"[{idx+1}/{len(records)}] Intent: {intent}")
        subtask_responses = []

        for j, subtask in enumerate(subtasks):
            print(f"    Subtask {j+1}: {subtask}")

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": subtask}],
                    temperature=0.7,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=1
                )
                reply = response.choices[0].message.content.strip()

            except Exception as e:
                reply = f"[ERROR]: {e}"

            subtask_responses.append({
                "subtask": subtask,
                "response": reply
            })

            time.sleep(0.5)  

        
        output_record = {
            "intent": intent,
            "subtask_responses": subtask_responses
        }

        out_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
        out_file.flush()
