TASK="ner"
MODEL="xlm-roberta-large"

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    for CKPT in "last" "oracle" "avg-ckpt"
    do
        sbatch -p gpu_4 -t 05:00:00 cumulative_averaging.sh $TASK $MODEL $CKPT $SEED 10 0
    done
done
