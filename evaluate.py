"""测评入口：拿一个（可选的）LoRA checkpoint 在 dev/test 上跑生成式测评。

用法：

    python evaluate.py                                    # 用默认配置测 dev+test
    python evaluate.py args/arg1.json                     # 指定配置
    python evaluate.py args/arg1.json checkpoint/best     # 指定 LoRA checkpoint
    python evaluate.py args/arg1.json checkpoint/best --splits test   # 只测 test

产出（默认写到 <save_dir>/eval/ 下）：

    predictions_dev.jsonl / predictions_test.jsonl   逐样本的 gold、pred、tp/fp/fn
    metrics.json                                     val_*/test_* 的 P/R/F1 与 per_type 明细

只想对已有预测文件重新打分，可以直接用：

    python metrics.py --pred <save_dir>/eval/predictions_test.jsonl --gold data/sft/test_sft.jsonl
"""

import argparse
import os

from train import DEFAULT_CONFIG, Trainer
from utils import Arguments


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounded MNER 测评")
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="配置文件路径（json）")
    parser.add_argument("checkpoint", nargs="?", default=None, help="LoRA checkpoint 目录，不填则用配置里的基座模型")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], help="要测评的 split：dev / test")
    parser.add_argument("--out", default=None, help="结果输出目录，默认 <save_dir>/eval")
    parser.add_argument("--batch-size", type=int, default=None, help="测评 batch size，默认用配置里的 batch_size")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="覆盖配置里的 max_new_tokens")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise SystemExit(f"配置文件不存在: {args.config}")
    if args.checkpoint and not os.path.exists(args.checkpoint):
        raise SystemExit(f"checkpoint 不存在: {args.checkpoint}")

    config = Arguments(args.config)
    if args.batch_size:
        config.eval_batch_size = args.batch_size
    if args.max_new_tokens:
        config.max_new_tokens = args.max_new_tokens

    trainer = Trainer(config, adapter_path=args.checkpoint)
    save_dir = args.out or (os.path.join(config.save_dir, "eval") if config.save_dir else None)
    metrics = trainer.evaluate(splits=tuple(args.splits), save_dir=save_dir)

    # 各 split 的汇总行也已经由 eval() 打印过，这里只做一次最终确认
    for split in args.splits:
        prefix = "test" if split == "test" else "val"
        if f"{prefix}_summary" in metrics:
            print(f"{split}: {metrics[f'{prefix}_summary']}")


if __name__ == "__main__":
    main()
