DEVICE=$1

for SEED in 1 2 3
do
    bash scripts/cocoop/xd_test.sh imagenetv2 ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh imagenet_sketch ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh imagenet_a ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh imagenet_r ${SEED} ${DEVICE}
done