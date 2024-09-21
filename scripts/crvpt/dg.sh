DEVICE=$1

for SEED in 1 2 3
do
    bash scripts/crvpt/xd_test_crvpt.sh imagenetv2 ${SEED} ${DEVICE}
    bash scripts/crvpt/xd_test_crvpt.sh imagenet_sketch ${SEED} ${DEVICE}
    bash scripts/crvpt/xd_test_crvpt.sh imagenet_a ${SEED} ${DEVICE}
    bash scripts/crvpt/xd_test_crvpt.sh imagenet_r ${SEED} ${DEVICE}
done