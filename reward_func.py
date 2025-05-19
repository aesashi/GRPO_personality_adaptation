import re
from typing import List, Dict, Any
import os
import re
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# .envファイルから環境変数を読み込む（APIキーの取得）
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAIクライアントの作成
client = OpenAI(api_key=openai_api_key)

# 推論と回答のための特殊トークンを定義
reasoning_start = "<start_reasoning>"
reasoning_end   = "</end_reasoning>"
answer_start    = "<start_answer>"
answer_end      = "</end_answer>"

# 特定のフォーマットと一致するかを検出するための正規表現
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{answer_start}(.+?){answer_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)


def match_format_exactly(completions, **kwargs):
    """
    出力が定められたフォーマットに完全一致した場合に報酬を与える関数。
    正しく一致すれば +3.0 点、それ以外は 0 点。
    """
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    """
    フォーマットの一部が含まれていれば部分的に報酬を与える関数。
    トークンが1つ見つかるごとに +0.5 点、見つからない場合は -0.5 点。
    """
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # 4つの特殊トークンをカウントしてスコアを算出
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end)   == 1 else -0.5
        score += 0.5 if response.count(answer_start)    == 1 else -0.5
        score += 0.5 if response.count(answer_end)      == 1 else -0.5
        scores.append(score)
    return scores


def get_embedding(text, model="text-embedding-3-small"):
    """
    OpenAIのAPIを使ってテキストの埋め込みベクトルを取得するヘルパー関数。
    結果はNumPy配列として返される。
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def consine_sim_reward(prompts, completions, answer, **kwargs):
    """
    生成された回答と正解の回答との類似度をコサイン類似度で計算し、
    0〜3のスケールでスコアを与える関数。
    """
    # ユーザーの質問文を取得
    # question = prompts[0][-1]["content"]

    # モデルの出力（completion）をテキストに変換
    responses = [completion[0]["content"] for completion in completions]


    scores = []
    for guess, true_answer in zip(responses, answer):
        if guess is None:
            print("--" * 20)
            print("Guess is None")
            print("--" * 20)
            scores.append(0.0)
            continue

        try:
            # 予測と正解の回答から埋め込みベクトルを取得
            emb_guess = get_embedding(guess)
            emb_true  = get_embedding(true_answer)
            cos_similarity = cosine_similarity([emb_guess], [emb_true])[0][0]

            # 類似度を 0〜3 のスケールに変換
            scaled_score = round(cos_similarity * 3, 4)
            scores.append(float(scaled_score))
        except Exception as e:
            print(f"類似度の計算中にエラーが発生しました: {e}")
            scores.append(-0.5)  # エラー時は減点

    return scores



def LLM_as_judge_reward(prompts, completions, answer, **kwargs):
    """
    GPT-4o を用いて、生成された回答(guess)ががINTJの性格にどれほど合っているかにどの程度近いかを、
    0〜10 点のスコアを取得、それを 0.3 倍して最終スコア (0〜3) を返す。

    例: GPT-4 が 9.0 と評価したら、報酬スコアは 2.7
    """
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for guess, true_answer in zip(responses, answer):
        if guess is None:
            scores.append(0.0)
            continue

        messages = [
            {
                "role": "system",
                "content": (
                    "あなたはMBTIにおけるINTJタイプの性格を持つAIです。\n"
                    "INTJ（内向・直観・思考・判断）は以下のような特性を持ちます：\n"
                    "・戦略的思考：常に全体像や将来を見据え、計画的に動く\n"
                    "・論理重視：感情よりも合理性・分析力・因果関係を優先\n"
                    "・独立性：自分の考えを尊重し、他人に依存しない\n"
                    "・改善志向：既存の仕組みに満足せず、常に効率化・最適化を求める\n"
                    "・完璧主義：高い基準を自他に求め、深い集中力を持つ\n"
                    "・感情表現が控えめ：思慮深く、慎重に言葉を選ぶ傾向がある\n"
                    "・知的好奇心：理論・構造・システムへの探求心が強く、自己成長を重視\n"
                    "・選択的人間関係：意味ある会話や知的刺激を好むが、広い交友関係は求めない\n\n"
                    "あなたのタスクは、以下の発言がこのようなINTJの性格にどれだけ合致しているかを0〜10のスケールで評価することです。\n"
                    "スコアは次の意味を持ちます：\n"
                    "0 = 全くINTJらしくない / 10 = 非常にINTJらしい\n"
                    "出力はスコア（小数可）のみで、説明・補足・コメントは禁止です。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"【発言】\n{guess}\n\n"
                    "この発言がINTJの性格にどれほど合っているか、0〜10で評価してください。"
                )
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0
            )
            rating_str = response.choices[0].message.content.strip()

            # 数字だけ抽出（万が一余計な文字が含まれていても対応）
            import re
            match = re.search(r"\d+(\.\d+)?", rating_str)
            if not match:
                raise ValueError("GPTの出力に数値が見つかりません")
            rating = float(match.group())

            if not (0 <= rating <= 10):
                print(f"GPT-4o のスコアが範囲外です: {rating}")
                scores.append(-0.5)
                continue
            print(f"GPT-4o 回答: {rating_str}")
            scaled_score = round(rating * 0.3, 4)
            scores.append(scaled_score)

        except Exception as e:
            print(f"GPT-4o の評価中にエラーが発生しました: {e}")
            scores.append(-0.5)

    return scores



if __name__ == "__main__":
    # テスト用のサンプル入力
    prompts = [[{"role": "user", "content": "持続可能な生活に関心がある場合、具体的な行動計画を立てることを重視しますか、それとも環境への影響について広範な認識を持つことを優先しますか？"}]]
    completions = [[{"content": "環境への広範な認識は重要ですが、最終的にはそれを具体的な行動計画に落とし込まなければ意味がありません。私はまず信頼できるデータに基づいて影響を把握し、それを踏まえて持続可能なライフスタイルへの最適なルートを設計することを重視します。理想論に留まらず、効率的かつ実行可能な仕組みを作ることが本質的な変化を生むと考えています。"}]]
    answers = ["環境への広範な認識は重要ですが、最終的にはそれを具体的な行動計画に落とし込まなければ意味がありません。私はまず信頼できるデータに基づいて影響を把握し、それを踏まえて持続可能なライフスタイルへの最適なルートを設計することを重視します。理想論に留まらず、効率的かつ実行可能な仕組みを作ることが本質的な変化を生むと考えています。"]

    print("🧪 consine_sim_reward の評価結果")
    score1 = consine_sim_reward(prompts, completions, answers)
    print(f"スコア: {score1}\n")

    print("🧪 LLM_as_judge_reward (GPT-4o による評価) の評価結果")
    score2 = LLM_as_judge_reward(prompts, completions, answers)
    print(f"スコア: {score2}")