DEVICE=$1

for SEED in 1 2 3
do
    bash scripts/language-prompting/xd_test_lp.sh imagenetv2 ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh imagenet_sketch ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh imagenet_a ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh imagenet_r ${SEED} ${DEVICE}
done