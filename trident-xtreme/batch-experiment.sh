TASK="nli" # or "tydiqa" "ner"
MODEL="xlm-roberta-large"
TIME="10:00:00" # time and infra specific, NLI max 16 hours for bs=16 on V100

# For NLI and TyDiQA, make sure to run ft-clf.sh and then create_clf_symlink.py first

for _ in 1 2 3
do
    for LR in 0.000001 0.000005 0.00001 0.000015 0.00002 0.000025 0.00003
    do
        for BS in 16 32 64
        do
            sbatch -p gpu_8 -t $TIME experiment.sh $TASK $MODEL $BS $LR  
            sbatch -p gpu_8 -t $TIME experiment.sh $TASK $MODEL $BS $LR  
            sbatch -p gpu_8 -t $TIME experiment.sh $TASK $MODEL $BS $LR  
        done
    done
done
