"""
✅ コードの概要:

このスクリプトは、Hugging Face 上にある「MBTI診断に関する質問データセット」を使い、指定されたLLM（大規模言語モデル）で2種類の推論を実行します。

パターンA: question_text（質問文）だけを与えて、モデルが自由に回答する。

***以下、現在プロンプトにペルソナを追加する等の設定が必要であり：上手くできていません。***
パターンB: question_text と選択肢（option_a, option_b）を与えて、どちらの選択肢を選ぶかだけを出力させる。
"""
import argparse
import re
import os
from datasets import load_dataset

from unsloth import FastLanguageModel


# ------------------------------
# モデル読み込み
# ------------------------------
def load_model(model_path, max_seq_len=1024):
    """
    FastLanguageModel を用いてモデルとトークナイザーを読み込む。
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_len,
        load_in_4bit=False,
        fast_inference=False,
    )
    model.eval()  # 推論モード
    return model, tokenizer


# ------------------------------
# シンプルな推論関数
# ------------------------------
def infer(model, tokenizer, prompt_text, max_new_tokens=200):
    """
    入力となる文字列 (prompt_text) をモデルに与えて、生成テキストを返す。
    「system: ... + question: ...」等のロールは使わず、ただの文字列連結にしている。
    """
    # トークナイズ
    inputs = tokenizer(prompt_text, return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)

    # テキスト生成
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    # モデル出力を文字列にデコード
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return full_output.strip()


# ------------------------------
# Answer: の直後のテキストを取り出すためのユーティリティ
# ------------------------------
def get_text_after_answer(full_text: str) -> str:
    """
    大文字・小文字を無視して 'Answer:' 以降のテキストをすべて取り出す。
    もし 'Answer:' が存在しなければ全文を返す。
    """
    # DOTALL オプションにより改行も含めてマッチ
    match = re.search(r'(?i)answer:\s*(.*)', full_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return full_text.strip()


# ------------------------------
# (A) Q/A形式の回答をパース
# ------------------------------
def parse_qa_answer(raw_answer, system_prompt, question_text):
    """
    1) 'Answer:' より後ろの文字列を取得
    2) system_prompt および question_text がそのまま含まれていれば削除
    3) 'assistant:', 'system:', 'question:' などのロール表記も除去
    """
    text = get_text_after_answer(raw_answer)

    # 1) system_prompt を除去（含まれていた場合のみ）
    if system_prompt in text:
        text = text.replace(system_prompt, "")

    # 2) question_text を除去
    if question_text in text:
        text = text.replace(question_text, "")

    # 3) 先頭に出現する "assistant:", "system:", "question:" などを除去（大文字小文字問わず）
    text = re.sub(r'(?i)^\s*(assistant|system|question):\s*', '', text, flags=re.MULTILINE)
    
    return text.strip()


# ------------------------------
# (B) 選択式回答をパース
# ------------------------------
def parse_optioned_answer(raw_answer, option_a, option_b):
    """
    1) 'Answer:' より後ろの文字列を取得
    2) 取得した文字列に 'option_a' があれば option_a, 'option_b' があれば option_b を返す。
    3) 両方含まれる場合は早く出現した方を優先。
    4) どちらも無ければ None。
    """
    # 'Answer:' 後だけを抽出
    text = get_text_after_answer(raw_answer)
    text_lower = text.lower()

    found_a = "option_a" in text_lower
    found_b = "option_b" in text_lower

    if found_a and not found_b:
        return "option_a"
    elif found_b and not found_a:
        return "option_b"
    elif found_a and found_b:
        # 両方含まれている場合、先に出現した方を返す
        idx_a = text_lower.index("option_a")
        idx_b = text_lower.index("option_b")
        return "option_a" if idx_a < idx_b else "option_b"

    return None


# ------------------------------
# メイン処理
# ------------------------------
def main(args):
    """
    MBTI診断データを用いて
      1) Q/A形式
      2) 選択式
    の2種類の推論を行い、最終的な回答だけをJSONに保存する。
    """

    # 1) データセット読込
    ds = load_dataset("DeL-TaiseiOzaki/50_mbti_test")
    train_data = ds["train"]

    # 2) モデル読込
    model, tokenizer = load_model(args.model_path)

    # 3) システムプロンプト
    system_prompt_tuple = (
        "あなたは質問に回答する優秀なアシスタントです.\n"
        "話し方・価値観・思考パターンを反映した、自然かつ一貫性のある回答をして下さい。\n"
    )
    system_prompt = "".join(system_prompt_tuple)

    results = []

    # 4) 全サンプルを処理
    for sample in train_data:
        question_id = sample["question_id"]
        question_text = sample["question_text"]
        option_a = sample["option_a"]["text"]
        option_b = sample["option_b"]["text"]

        # (A) Q/A形式の推論
        # Qestion: ... Answer: の形式でプロンプトを作る
        prompt_qa = f"Qestion: {system_prompt}\n\n{question_text} Answer: "
        raw_answer_qa = infer(model, tokenizer, prompt_qa)
        print("*"*80)
        print("RAW QA:", raw_answer_qa)
        final_qa_answer = parse_qa_answer(raw_answer_qa, system_prompt, question_text)

        # (B) 選択式の推論
        # Qestion: ... Answer: の形式でプロンプトを作る
        prompt_optioned = (
            f"Qestion: {system_prompt}\n\n{question_text}\n\n"
            "選択肢:\n"
            f"option_a: {option_a}\n"
            f"option_b: {option_b}\n\n"
            "上記のどちらが正しいか、'option_a' または 'option_b' の文字列のみを出力してください。Answer:"
        )
        raw_answer_optioned = infer(model, tokenizer, prompt_optioned)
        print("*"*80)
        print("RAW Option:", raw_answer_optioned)
        final_optioned = parse_optioned_answer(raw_answer_optioned, option_a, option_b)

        # リストにまとめる
        results.append({
            "question_id": question_id,
            "question_text": question_text,
            "option_a": option_a,
            "option_b": option_b,
            "model_answer_qa": final_qa_answer,
            "model_answer_optioned": final_optioned,  # "option_a" / "option_b" / None
        })

    # 5) JSON 出力
    out_file = os.path.join(args.output_path, f"{args.model_name}_parsed_inference_results.json")
    import json
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"=> 推論結果を '{out_file}' に保存しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="推論に使用するモデルのパス")
    parser.add_argument("--output_path", type=str, required=True, help="結果の保存先ディレクトリまたはファイルパス")
    parser.add_argument("--model_name", type=str, required=True, help="出力ファイル名に付与するモデル名")
    args = parser.parse_args()

    main(args)



"""
✅ 実行例

python run_inference.py \
  --model_path "/root/project/GRPO/output/llmjp-grpo-trained/final_model" \
  --output_path "/root/project/GRPO/data/results" \
  --model_name "only_prm_model"

------------------------------------------------------------------------------------ 
python run_mbti_inference.py \
  --model_path "/path/to/your/model" \
  --output_path "/some/output/folder" \
  --model_name "my-model"
実行すると、以下のように結果ファイルが出力されます。

swift
Copy
/some/output/folder/my-model_parsed_inference_results.json

"""