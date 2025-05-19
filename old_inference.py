import json
from unsloth import FastLanguageModel
from transformers import pipeline
from tqdm import tqdm
import argparse
import re

# ---- For similarity ----
from sentence_transformers import SentenceTransformer, util

# ------------------------------
# モデルとトークナイザーの読み込み
# ------------------------------
def load_model(model_path, max_seq_len=1024):
    """Load the FastLanguageModel and tokenizer."""
    # fast_inference=False に設定し、安全に動作させる
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_len,
        load_in_4bit=False,
        fast_inference=False,
    )
    model.eval()  # 推論モード
    return model, tokenizer

# ------------------------------
# 文章埋め込みモデルの読み込み
# ------------------------------
def load_similarity_model():
    # Choose any SentenceTransformer model from:
    # https://www.sbert.net/docs/pretrained_models.html
    similarity_model = SentenceTransformer("all-mpnet-base-v2")
    return similarity_model

# ------------------------------
# プロンプト構築
# ------------------------------
def build_prompt(messages):
    """Convert role/content messages into a single prompt string."""
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

# ------------------------------
# 推論処理
# ------------------------------
def infer(model, tokenizer, question, system_prompt):
    """Generate a response from the model given the question and system prompt."""
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    input_text = build_prompt(prompt)

    # トークナイズ
    inputs = tokenizer(input_text, return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)

    # 推論（テキスト生成）
    output_ids = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # "Assistant:" 以降だけを抽出（最初に見つかった箇所）
    match = re.search(r"Assistant:\s*(.*)", full_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # "Assistant:" が見つからない場合は全文を返す
        return full_output.strip()

# ------------------------------
# コサイン類似度を計算
# ------------------------------
def compute_similarity(reference_text, inference_text, similarity_model):
    if not reference_text or not inference_text:
        return None
    ref_emb = similarity_model.encode(reference_text, convert_to_tensor=True)
    inf_emb = similarity_model.encode(inference_text, convert_to_tensor=True)
    score = util.cos_sim(ref_emb, inf_emb).item()
    return score

# ------------------------------
# メイン処理
# ------------------------------
def main(args):
    data_path = "/root/project/GRPO-Ozaki/data/results/answers_ENFP_nemotron_with_results.json"

    # システムプロンプト
    system_prompt = """あなたは質問に回答する優秀なアシスタントです.
話し方・価値観・思考パターンを反映した、自然かつ一貫性のある回答をして下さい。"""

    # JSONデータの読み込み
    with open(data_path, "r", encoding="utf-8") as f:
        file_content = json.load(f)

    # もし top-level が dict で {"data": [...], "cosine_similarity": {...}} 等なら抽出
    # そうでなければ list の想定
    if isinstance(file_content, dict):
        data = file_content.get("data", [])
        cosine_info = file_content.get("cosine_similarity", {})
    else:
        # 単純に list が top-level の場合
        data = file_content
        # 後ほど集計結果を格納するため dict にする
        file_content = {"data": data, "cosine_similarity": {}}
        cosine_info = file_content["cosine_similarity"]

    # モデル読み込み
    model, tokenizer = load_model(args.model_path)
    similarity_model = load_similarity_model()

    # 類似度集計用
    sum_sim = 0.0
    count_sim = 0

    # 指定数のアイテムに対して推論
    for i in tqdm(range(min(args.num_samples, len(data))), desc="推論中"):
        item = data[i]
        question = item.get("generated_question", "")
        if not question:
            continue  # 質問が空ならスキップ

        # 既に output_key が存在する場合は上書きしない
        if args.output_key not in item:
            # 推論を実行
            response = infer(model, tokenizer, question, system_prompt)
            item[args.output_key] = response

            # 既存の "answer" と類似度計算
            reference_answer = item.get("answer", "")
            similarity_score = compute_similarity(reference_answer, response, similarity_model)

            # スコアを保存
            item[args.output_key + "_similarity"] = similarity_score
            # 平均計算のため累積
            if similarity_score is not None:
                sum_sim += similarity_score
                count_sim += 1

    # 最後に集計した平均類似度を top-level の "cosine_similarity" に記録
    avg_sim = sum_sim / count_sim if count_sim > 0 else 0.0
    # args.output_key がモデル名扱いになるので、その名前で保存
    cosine_info[args.output_key] = avg_sim

    # 更新した data と 集計結果を file_content に格納
    file_content["data"] = data
    file_content["cosine_similarity"] = cosine_info

    # JSONファイルに上書き保存
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(file_content, f, ensure_ascii=False, indent=2)

    print(f"推論・類似度計算完了: {data_path}")
    print(f"モデル '{args.output_key}' の平均類似度: {avg_sim}")

# ------------------------------
# コマンドライン引数
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定したモデルで推論を実行し、結果と平均類似度を元のJSONファイルに追加します。")
    parser.add_argument("--model_path", type=str, required=True, help="使用するモデルのディレクトリパス")
    parser.add_argument("--output_key", type=str, required=True, help="JSONに追加する出力キー名 (例: model1_output)")
    parser.add_argument("--num_samples", type=int, default=100, help="処理する質問数（デフォルト: 100）")
    args = parser.parse_args()

    main(args)
