#!/bin/zsh

#SBATCH --mail-user=t.shamoyan@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END
#SBATCH --job-name=train_on_external_valtest_on_yamanishi
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a3090
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --exclude=devbox2
#SBATCH --output=%x.%j.out.log
#SBATCH --error=%x.%j.err.log

# ---------- Parameters ----------
DATASET_ROOT="data_k_fold/train_on_external_valtest_on_yamanishi"
DATASET_TYPE="ion_channel"
DATASET_NAME="with_side_effects"
FOLD=1
DATA_PATH="${DATASET_ROOT}/${DATASET_TYPE}/${DATASET_NAME}/${FOLD}"

MODEL_KGE="DistMult"                # TransE | RotatE | ComplEx | DistMult

HP=`apptainer exec /nfs/data/env/jq.sif jq ".${DATASET_TYPE}.${MODEL_KGE}" hyperparameters.json`

NEGATIVES=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".NEGATIVES" -r)
BATCHSIZE=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".BATCHSIZE" -r)
DIM=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".DIM" -r)
GAMMA=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".GAMMA" -r)
LR=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".LR" -r)
MAX_STEPS=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".MAX_STEPS" -r)
VAL_STEPS=$(expr $MAX_STEPS / 20)
DE=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".DE" -r)
DR=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".DR" -r)
REG=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".REG" -r)

NEG_SAMPLING_METHOD="uniform"               # [ uniform ] | jaccard | count
EVAL_NEG_SAMPLING_METHOD="jaccard"          # [ jaccard ] | uniform | count | type

WANDB_ENTITY="l3s-future-lab"
WANDB_PROJECT="train_on_external_valtest_on_yamanishi"
WANDB_RUN_NAME="${DATASET_TYPE}_${DATASET_NAME}_${MODEL_KGE}_eval-${EVAL_NEG_SAMPLING_METHOD}"

# --------- Load Modules ---------
source ~/.zshrc
conda activate kge && "$@"

# ----- Export Env Variables -----
source .env
export WANDB_API_KEY=$wandb_api_key

# ------------- Run --------------
python -u codes/run.py \
 --wandb_project $WANDB_PROJECT \
 --wandb_entity $WANDB_ENTITY \
 --wandb_run_name $WANDB_RUN_NAME \
 --neg_sampling_method $NEG_SAMPLING_METHOD \
 --eval_neg_sampling_method $EVAL_NEG_SAMPLING_METHOD \
 --do_train \
 --cuda \
 --seed 42 \
 --do_valid \
 --do_test \
 --data_path $DATA_PATH \
 --model $MODEL_KGE \
 --valid_steps $VAL_STEPS \
 --log_steps $VAL_STEPS \
 -n $NEGATIVES -b $BATCHSIZE -d $DIM \
 -g $GAMMA -a 1.0 -adv \
 -lr $LR --max_steps $MAX_STEPS \
 -save models/"${DATASET_TYPE}_${DATASET_NAME}_${MODEL_KGE}" \
 --test_batch_size 8 $DE $DR $REG
