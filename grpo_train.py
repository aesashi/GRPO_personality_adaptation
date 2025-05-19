import argparse
import json
import os
import re
from typing import Dict, List, Any

import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

# === 1) reward.py からカスタム報酬関数をインポート ===
from reward_func import (
    match_format_exactly,
    match_format_approximately,
    consine_sim_reward,
    LLM_as_judge_reward,
)

import wandb


def parse_args():
    # コマンドライン引数のパーサーを定義
    parser = argparse.ArgumentParser(description="カスタム報酬関数を用いたGRPOトレーニング")
    parser.add_argument("--model_name", type=str, required=True, help="読み込む事前学習済みモデルの名前")
    parser.add_argument("--data_path", type=str, required=True, help="JSONまたはJSONLデータファイルのパス")
    parser.add_argument("--output_dir", type=str, required=True, help="学習済みモデルの保存先ディレクトリ")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="最大シーケンス長")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRAのランク")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学習率")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="学習エポック数")
    parser.add_argument("--max_steps", type=int, default=1000, help="最大学習ステップ数")
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="勾配累積ステップ数")
    parser.add_argument("--num_generations", type=int, default=8, help="プロンプトごとの生成回数")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPUメモリの使用率")
    return parser.parse_args()

import json
from datasets import Dataset

def load_data(data_path: str) -> Dataset:
    """
    Load and process dataset for GRPO training.
    Requires each data entry to have 'generated_question' and 'answer' fields.
    Returns a Hugging Face Dataset with 'question', 'answer', and 'prompt'.
    """
    # Load JSON or JSONL
    if data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = [json.loads(line) for line in f]

    # Define system prompt
    system_prompt = (
        "あなたは質問に回答する優秀なアシスタントです.\n"
        "話し方・価値観・思考パターンを反映した、自然かつ一貫性のある回答をして下さい。\n"
    )

    # Create Hugging Face Dataset
    dataset = Dataset.from_list(raw_data)

    # Format each entry with required fields
    dataset = dataset.map(lambda x: {
        "question": x["generated_question"],
        "answer": x["answer"],
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["generated_question"]},
        ]
    })

    # Retain only required columns
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ['question', 'answer', 'prompt']]
    )

    return dataset

def main():
    # コマンドライン引数の取得
    args = parse_args()
    wandb.init(
        project="GRPO-INTJ-Cosine-Sim-and-LLM-as-Judge",          
        config=vars(args)                
    )
    # モデルとトークナイザーの読み込みと初期化
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # LoRAによるモデルの微調整設定
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # データの読み込み
    dataset = load_data(args.data_path)

    # トレーニングパラメータの設定
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=512,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.max_steps // 5,
        max_grad_norm=0.1,
        report_to = "wandb",
        output_dir=args.output_dir,
    )

    # GRPOTrainerの初期化（報酬関数を指定）
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            consine_sim_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # トレーニングの開始
    print(f"{len(dataset)} 件のデータでGRPOトレーニングを開始します")
    trainer.train()

    # 学習済みモデルの保存
    print("トレーニング完了。モデルを保存中...")
    final_model_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"モデルを {final_model_dir} に保存しました")

    wandb.finish()

if __name__ == "__main__":
    main()
