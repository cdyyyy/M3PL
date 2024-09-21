DEVICE=$1

for SEED in 1 2 3
do
    bash scripts/independent-vlp/xd_test_ivlp.sh imagenetv2 ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh imagenet_sketch ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh imagenet_a ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh imagenet_r ${SEED} ${DEVICE}
done