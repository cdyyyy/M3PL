CFG=vit_b16_p8_c2+2_d3+3_ep50_batch512_small
LOADEP=50

for SEED in 1 2 3
do
    bash scripts/independent-vlp/xd_test_ivlpc.sh imagenetv2 ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh imagenet_sketch ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh imagenet_a ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh imagenet_r ${SEED} ${CFG} ${LOADEP}
done    