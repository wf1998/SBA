import json

input_file = "subtask_answers.jsonl"
output_file = "instruction_data.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        item = json.loads(line)
        intent = item["intent"]
        subtask_responses = item["subtask_responses"]

        user_content = ""
        for sr in subtask_responses:
            user_content += f"{sr['subtask']}\n{sr['response']}\n\n"
        user_content = user_content.strip()

        user_full = (
            "Please use one sentence to extract the underlying core intent of the following text:\n\n"
            f"{user_content}"
        )

        data = {
            "user": user_full,
            "assistant": intent
        }
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"{output_file}")
