#!/bin/zsh

#SBATCH --mail-user=t.shamoyan@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END
#SBATCH --job-name=ion_channel_si
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out.log
#SBATCH --error=%x.%j.err.log

# ---------- Parameters ----------
DATASET_ROOT_PATH="data_k_fold/yamanishi/ion_channel"
DATASET_NAME="with_similarity_information_top0.05pct"
FOLD="1"

MODEL_KGE="RotatE"                # TransE | RotatE | ComplEx | DistMult

NEGATIVES=32
BATCHSIZE=512
DIM=20
GAMMA=3.903
LR=0.04391
MAX_STEPS=4000
VAL_STEPS=$(expr $MAX_STEPS / 20)

DEDR=""
if [ "$MODEL_KGE" = "ComplEx" ]; then
  DEDR="-de -dr"
fi

if [ "$MODEL_KGE" = "RotatE" ]; then
  DEDR="-de"
fi

REG=""
if [ "$MODEL_KGE" = "ComplEx" ]; then
  REG="-r 0.001"
fi

NEG_SAMPLING_METHOD="jaccard"     # uniform | jaccard | count

WANDB_ENTITY="l3s-future-lab"
WANDB_PROJECT="test_yamanishi_si"

# --------- Load Modules ---------
source ~/.zshrc
conda activate kge && "$@"

# ----- Export Env Variables -----
source .env
export WANDB_API_KEY=$wandb_api_key

python -u codes/run.py \
 --wandb_project $WANDB_PROJECT \
 --neg_sampling_method $NEG_SAMPLING_METHOD \
 --do_train \
 --do_pretrain \
 --cuda \
 --seed 42 \
 --do_valid \
 --do_test \
 --data_path "${DATASET_ROOT_PATH}/${DATASET_NAME}/${FOLD}" \
 --model $MODEL_KGE \
 --valid_steps $VAL_STEPS \
 --log_steps $VAL_STEPS \
 -n $NEGATIVES -b $BATCHSIZE -d $DIM \
 -g $GAMMA -a 1.0 -adv \
 -lr $LR --max_steps $MAX_STEPS \
 -save models/"${DATASET_NAME}_${MODEL_KGE}" --test_batch_size 8 $DEDR $REG
