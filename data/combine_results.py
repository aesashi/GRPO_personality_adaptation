import os
import json
import tiktoken

def main():
    results_dir = "./results"
    questions_map = {}

    # Tokenizer setup
    def count_tokens(text, model_name="gpt-3.5-turbo"):
        if not text:
            return 0
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))

    # For storing token statistics per file
    file_token_stats = {}  # { filename: {"total_tokens": int, "count": int} }

    json_files = [
        f for f in os.listdir(results_dir)
        if f.endswith(".json")
    ]

    for json_file in json_files:
        file_path = os.path.join(results_dir, json_file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            qid = entry["question_id"]

            if qid not in questions_map:
                questions_map[qid] = {
                    "question_id": qid,
                    "question_text": entry["question_text"],
                    "option_a": entry["option_a"],
                    "option_b": entry["option_b"],
                    "answers": {}
                }

            answer_qa = entry.get("model_answer_qa", "")
            answer_optioned = entry.get("model_answer_optioned", "")

            answer_qa_tokens = count_tokens(answer_qa)

            questions_map[qid]["answers"][json_file] = {
                "model_answer_qa": answer_qa,
                "model_answer_qa_tokens": answer_qa_tokens,
                "model_answer_optioned": answer_optioned,
            }

            # Update file-based QA token statistics
            if json_file not in file_token_stats:
                file_token_stats[json_file] = {"total_tokens": 0, "count": 0}
            file_token_stats[json_file]["total_tokens"] += answer_qa_tokens
            file_token_stats[json_file]["count"] += 1

    combined_data = list(questions_map.values())
    combined_data.sort(key=lambda x: x["question_id"])

    output_file = "./compare_inferences/combined_results.json"
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(combined_data, out, ensure_ascii=False, indent=2)

    print(f"結合された結果を '{output_file}' に出力しました。\n")

    # Display average QA token count per file
    print("ファイルごとの model_answer_qa の平均トークン数:")
    for filename, stats in file_token_stats.items():
        avg_tokens = stats["total_tokens"] / stats["count"] if stats["count"] else 0
        print(f"  {filename}: 平均 {avg_tokens:.2f} トークン")

if __name__ == "__main__":
    main()
