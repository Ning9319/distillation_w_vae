GPU=$1
DATA=$2
LR=$3
IPC=$4


CUDA_VISIBLE_DEVICES=${GPU} python baseline.py \
--method DM \
--dataset ${DATA} \
--ipc ${IPC} \
--num_eval 1 \
--epoch_eval_train 500 \
--init real \
--lr_img ${LR} \
--lr_net 0.01 \
--Iteration 5000 \
--model ConvNet3D \
--eval_mode SS \
--eval_it 500 \
--batch_real 16 \
--num_workers 4 \
--preload \
