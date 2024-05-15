import argparse
import subprocess
import traceback
import sys
import wandb


def sweep_action():
    wandb.init()
    try:
        hparams = dict(wandb.config)
        subprocess.run(
            [
                "python3",
                "main.py",
                "fit",
                "-c",
                "configs/run-config-exp2.yaml",
                *[f"--{k}={v}" for k, v in hparams.items()],
                # f"--trainer.limit_train_batches={1_280_000 / hparams['data.init_args.train_batch_size']}",
                f"--data.init_args.pretrained_tokenizer_name_or_path={hparams['model.init_args.pretrained_model_name_or_path']}",
                f"--model.init_args.learning_rate={5e-5 / hparams['data.init_args.train_batch_size']}",
                f"--trainer.val_check_interval={1000 * 256 / hparams['data.init_args.train_batch_size']}",
            ]
        )
    except Exception:
        print(traceback.print_exc(), file=sys.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single wandb parameter sweep agent.")
    parser.add_argument("--sweep_id", required=True, help="The wand sweep ID to register with")
    args = parser.parse_args()
    wandb.agent(sweep_id=args.sweep_id, function=sweep_action)
