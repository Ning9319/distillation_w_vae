GPU=$1
DATA=$2
IPC=$3


CUDA_VISIBLE_DEVICES=${GPU} python distill_w_2dvae.py \
--dataset ${DATA} \
--ipc ${IPC} \
--num_eval 5 \
--epoch_eval_train 500 \
--init real \
--lr_net 0.01 \
--model ConvNet3D \
--eval_mode SS \
--num_workers 4 \
--preload \
