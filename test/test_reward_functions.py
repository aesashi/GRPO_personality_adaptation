import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import reward_func
from reward_func import consine_sim_reward, LLM_as_judge_reward

#-------------------------------------------------------------------------------------------------------------------
## reward_funcの関数をテストするためのモックを作成

# テスト用のデータを準備する fixture
@pytest.fixture
def sample_data():
    # ユーザープロンプト、モデル出力、正解データ
    prompts = [[{"role": "user", "content": "フランスの首都は何ですか？"}]]
    completions = [[{"content": "パリはフランスの首都です。なので答えはパリです。"}]]
    answers = ["パリ"]
    return prompts, completions, answers

# 完全一致した場合のテスト（類似度 1.0 → スコア 3.0）
@patch("reward_func.get_embedding")  
def test_consine_sim_reward_perfect_match(mock_get_embedding, sample_data):
    # 完全に同じ埋め込みベクトルを返すモック
    mock_get_embedding.side_effect = lambda text: np.array([1.0, 0.0])
    prompts, completions, answers = sample_data

    scores = consine_sim_reward(prompts, completions, answers)
    assert len(scores) == 1
    assert pytest.approx(scores[0], 0.01) == 3.0

# フォーマットが不正（タグが不足）の場合のテスト
@patch("reward_func.get_embedding")
def test_consine_sim_reward_missing_format(mock_get_embedding, sample_data):
    prompts, _, answers = sample_data
    completions = [[{"content": "パリ"}]]

    # 埋め込み取得を空のベクトルにする（意図的にエラーを起こす）
    mock_get_embedding.return_value = np.array([])

    scores = consine_sim_reward(prompts, completions, answers)

    # エラー時は -0.5 が返る設計なのでそれを期待
    assert scores == [-0.5]

# 部分的に一致した場合のテスト（類似度 0.5 → スコア 1.5）
@patch("reward_func.get_embedding")
def test_consine_sim_reward_partial_match(mock_get_embedding, sample_data):
    # 類似度が約 0.5 になるようなベクトルを返す
    emb_guess = np.array([1.0, 0.0])
    emb_true  = np.array([0.5, 0.866])  # 約60度の角度 → cos ≒ 0.5
    mock_get_embedding.side_effect = [emb_guess, emb_true]
    prompts, completions, answers = sample_data

    scores = consine_sim_reward(prompts, completions, answers)
    assert pytest.approx(scores[0], 0.1) == 1.5


#-------------------------------------------------------------------------------------------------------------------
## LLM_as_judge_reward関数をテストするためのモックを作成

import pytest
from unittest.mock import patch, MagicMock
from reward_func import LLM_as_judge_reward, consine_sim_reward
import numpy as np

@pytest.fixture
def sample_data():
    prompts = [[{"role": "user", "content": "東京は日本の首都ですか？"}]]
    completions = [[{"content": "はい、東京は日本の首都です。"}]]
    answers = ["はい、東京は日本の首都です。"]
    return prompts, completions, answers

# コサイン類似度テスト：埋め込みエラー時のハンドリング確認
@patch("reward_func.get_embedding")
def test_consine_sim_reward_invalid_embedding(mock_embed, sample_data):
    # 空ベクトルを返してエラーを起こさせる
    mock_embed.return_value = np.array([])
    prompts, completions, answers = sample_data
    scores = consine_sim_reward(prompts, completions, answers)
    assert scores == [-0.5]

# GPT評価: 正常ケース（10 -> 3.0）
@patch("reward_func.client.chat.completions.create")
def test_LLM_as_judge_reward_perfect(mock_chat, sample_data):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "10"
    mock_chat.return_value = mock_resp

    prompts, completions, answers = sample_data
    scores = LLM_as_judge_reward(prompts, completions, answers)
    assert pytest.approx(scores[0], 0.01) == 3.0

# GPT評価: 文字列混じり -> 正常に数値抽出できるか
@patch("reward_func.client.chat.completions.create")
def test_LLM_as_judge_reward_with_text(mock_chat, sample_data):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "これはスコアです: 7点"
    mock_chat.return_value = mock_resp

    prompts, completions, answers = sample_data
    scores = LLM_as_judge_reward(prompts, completions, answers)
    assert pytest.approx(scores[0], 0.1) == 2.1

# GPT評価: 範囲外の数値 -> -0.5
@patch("reward_func.client.chat.completions.create")
def test_LLM_as_judge_reward_out_of_range(mock_chat, sample_data):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "15"
    mock_chat.return_value = mock_resp

    prompts, completions, answers = sample_data
    scores = LLM_as_judge_reward(prompts, completions, answers)
    assert scores == [-0.5]

# GPT評価: 数値がない場合 -> -0.5
@patch("reward_func.client.chat.completions.create")
def test_LLM_as_judge_reward_no_number(mock_chat, sample_data):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "とても良い回答でした。"
    mock_chat.return_value = mock_resp

    prompts, completions, answers = sample_data
    scores = LLM_as_judge_reward(prompts, completions, answers)
    assert scores == [-0.5]
