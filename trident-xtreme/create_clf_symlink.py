import os
import glob

def create_symlink_last_ckpt(cwd):
    tasks = ["nli", "ner", "tydiqa"]
    models = ["xlm-roberta-large"]

    for task in tasks:
        for model in models:
            if task == "tydiqa":
                pattern = os.path.join(cwd, f"logs/ofa-head/{task}/{model}/batch_size=32/lr=2e-05/scheduler=0.1/seed=0/checkpoints/epoch=39_val=*.ckpt")
            else:
                pattern = os.path.join(cwd, f"logs/ofa-head/{task}/{model}/batch_size=32/lr=2e-05/scheduler=0.1/seed=0/checkpoints/epoch=9*_val=*.ckpt")
            files = glob.glob(pattern)
            if files:
                latest_checkpoint = max(files, key=os.path.getctime)
                symlink_path = os.path.join(os.path.dirname(latest_checkpoint), 'last.ckpt')
                if os.path.exists(symlink_path):
                    os.remove(symlink_path)
                os.symlink(latest_checkpoint, symlink_path)
                print(f"Created symlink from {latest_checkpoint} to {symlink_path}")
            else:
                print(f"No epoch=9 checkpoint found for {task}/{model}")

# Call the function with the current working directory (CWD)
create_symlink_last_ckpt(os.getcwd())

