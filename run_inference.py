"""
✅ コードの概要:

このスクリプトは、Hugging Face 上にある「MBTI診断に関する質問データセット」を使い、指定されたLLM（大規模言語モデル）で2種類の推論を実行します。

パターンA: question_text（質問文）だけを与えて、モデルが自由に回答する。

パターンB: question_text と選択肢（option_a, option_b）を与えて、どちらの選択肢を選ぶかだけを出力させる。

さらに、オプションで回答のコサイン類似度も計算可能です。
"""

import argparse
import re
from datasets import load_dataset
import json
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer, util


# ------------------------------
# モデルの読み込み関数
# ------------------------------
def load_model(model_path, max_seq_len=1024):
    """
    FastLanguageModel を使ってモデルとトークナイザーを読み込む。
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_len,
        load_in_4bit=False,
        fast_inference=False,
    )
    model.eval()  # 推論モードに設定
    return model, tokenizer


# ------------------------------
# プロンプトの組み立て関数
# ------------------------------
def build_prompt(messages):
    """
    system / user 形式のリスト（[{role:..., content:...}, ...]）を
    ChatGPT風のテキスト形式に変換する。
    """
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])


# ------------------------------
# 推論を実行する関数
# ------------------------------
def infer(model, tokenizer, system_prompt, user_message, max_new_tokens=200):
    """
    モデルに user_message を与え、生成された回答を文字列として返す。
    """
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    input_text = build_prompt(prompt)

    # トークナイズ
    inputs = tokenizer(input_text, return_tensors="pt")
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

    # 出力を文字列にデコード
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # "Assistant:" 以降を抽出する（先頭に出力されている想定の場合）
    match = re.search(r"Assistant:\s*(.*)", full_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return full_output.strip()


# ------------------------------
# 類似度モデルの読み込み
# ------------------------------
def load_similarity_model():
    """
    文章同士の類似度を測るための SentenceTransformer モデルを読み込む。
    """
    return SentenceTransformer("all-mpnet-base-v2")


# ------------------------------
# コサイン類似度を計算
# ------------------------------
def compute_similarity(text1, text2, similarity_model):
    """
    2つのテキストのコサイン類似度を計算して数値を返す。
    """
    if not text1 or not text2:
        return None
    emb1 = similarity_model.encode(text1, convert_to_tensor=True)
    emb2 = similarity_model.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return score


# ------------------------------
# 選択式回答をパースして "option_a" or "option_b" を抽出
# ------------------------------
def parse_optioned_answer(raw_answer, option_a, option_b, similarity_model=None):
    """
    モデル回答（raw_answer）が、必ずしも 'option_a' or 'option_b' だけを
    返さない場合に対応するための関数。
    1) 文字列に 'option_a' or 'option_b' が含まれていれば、そのまま返す。
    2) 含まれない場合は、option_a, option_b の内容と raw_answer の類似度を計算し、
       より近い方を選ぶ。
    """

    text_lower = raw_answer.lower()

    # 1) 回答文に "option_a" or "option_b" が含まれているか確認
    found_a = "option_a" in text_lower
    found_b = "option_b" in text_lower

    if found_a and not found_b:
        return "option_a"
    elif found_b and not found_a:
        return "option_b"
    elif found_a and found_b:
        # 両方含まれている場合 → どちらかを優先する必要がある
        # ここでは簡単に先に出現した方を返す例
        idx_a = text_lower.index("option_a")
        idx_b = text_lower.index("option_b")
        return "option_a" if idx_a < idx_b else "option_b"

    # 2) "option_a" / "option_b" が回答に含まれていない場合
    #    → 類似度を使って判定（オプション）
    if similarity_model is not None:
        # option_a / option_b のテキスト(例: "大人数と交流する" / "1対1で話す" )
        # と raw_answer の類似度を比較
        sim_a = compute_similarity(raw_answer, option_a, similarity_model)
        sim_b = compute_similarity(raw_answer, option_b, similarity_model)

        # どちらの類似度が高いかで判定
        if sim_a is not None and sim_b is not None:
            if sim_a > sim_b:
                return "option_a"
            else:
                return "option_b"

    # 類似度を使わない or どちらもスコア計算できなかった場合は、
    # 簡易的にキーワードを見る例
    # 例として "大人数" があれば option_a, "1対1" なら option_b など
    lower_a = option_a.lower()
    lower_b = option_b.lower()
    if any(keyword in text_lower for keyword in ["大人数", "みんなで", "みんなと"]):
        return "option_a"
    if any(keyword in text_lower for keyword in ["1対1", "一対一", "一人", "ひとり"]):
        return "option_b"

    # それでも判断できない場合は None
    return None


# ------------------------------
# メイン処理
# ------------------------------
def main(args):
    """
    新しいMBTI診断用データセットを用いて、2パターンの推論を行う。
    選択式回答は parse_optioned_answer() で最終的に 'option_a' or 'option_b' を抽出する。
    """

    # 1) データセットを読み込む
    ds = load_dataset("DeL-TaiseiOzaki/50_mbti_test")
    train_data = ds["train"]

    # 2) モデルと（必要なら）類似度モデルを読み込む
    model, tokenizer = load_model(args.model_path)
    similarity_model = load_similarity_model() if args.use_similarity else None

    # システムプロンプト
    system_prompt = (
        "あなたの性格どおりに質問に回答してください。\n",
        "話し方・価値観・思考パターンを反映した、自然かつ一貫性のある回答をして下さい。\n"
    )

    # 結果を格納するリスト
    results = []

    # 3) データをループ
    for i, sample in enumerate(train_data):
        if i >= args.num_samples:
            break

        question_id = sample["question_id"]
        question_text = sample["question_text"]
        option_a = sample["option_a"]["text"]
        option_b = sample["option_b"]["text"]

        # --- A) 質問をそのまま投げて自由回答を得る ---
        user_msg_qa = question_text
        answer_qa = infer(model, tokenizer, system_prompt, user_msg_qa)

        # --- B) 選択式: option_a / option_b を与えて答えさせる ---
        user_msg_optioned = (
            f"質問: {question_text}\n\n"
            f"選択肢:\n"
            f"option_a: {option_a}\n"
            f"option_b: {option_b}\n\n"
            "上記のどちらが正しいか、'option_a' または 'option_b' の文字列のみを出力してください。"
        )
        raw_answer_optioned = infer(model, tokenizer, system_prompt, user_msg_optioned)

        # 選択式回答をパースして "option_a" or "option_b" を特定
        parsed_option = parse_optioned_answer(raw_answer_optioned, option_a, option_b, similarity_model)

        # （オプション）何かの正解ラベルがあれば類似度を計算
        # 今回は例としてnullのままにしている
        reference_answer = None
        sim_qa = compute_similarity(reference_answer, answer_qa, similarity_model) if reference_answer else None
        sim_optioned = compute_similarity(reference_answer, raw_answer_optioned, similarity_model) if reference_answer else None

        # 結果を保存
        results.append({
            "question_id": question_id,
            "question_text": question_text,
            "option_a": option_a,
            "option_b": option_b,
            "model_answer_q/a": answer_qa,
            "similarity_a": sim_qa,
            "model_answer_optioned_raw": raw_answer_optioned,   # 生の回答（パース前）
            "model_answer_optioned_parsed": parsed_option,      # "option_a", "option_b", もしくは None
            "similarity_b": sim_optioned,
        })

    # 4) 出力例：最初の数件をコンソールに表示
    for item in results[:5]:
        print("-----")
        print(f"質問ID: {item['question_id']}")
        print(f"Q: {item['question_text']}")
        print(f"自由回答: {item['model_answer_q/a']}")
        print(f"選択式（生回答）: {item['model_answer_optioned_raw']}")
        print(f"選択式（パース後）: {item['model_answer_optioned_parsed']}")
        print("-----")

    with open("./data/results/inference_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("=> parsed_inference_results.json に結果を保存しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="推論に使用するモデルのパス")
    parser.add_argument("--num_samples", type=int, default=10, help="テストするデータ数")
    parser.add_argument("--use_similarity", action="store_true", help="回答のテキスト類似度を計算するか")
    args = parser.parse_args()

    main(args)



"""
✅ 実行例

python run_inference.py \
  --model_path "/root/project/GRPO/output/llmjp-grpo-trained/final_model" \
  --num_samples 8

  
類似度を有効にする場合：
python run_inference.py \
  --model_path "/path/to/your/model" \
  --num_samples 8 \
  --use_similarity

"""