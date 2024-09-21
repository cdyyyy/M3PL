CFG=vit_b16_p8_c2+2_d3+3_ep50_batch512_small
LOADEP=50

for SEED in 1 2 3
do
    bash scripts/independent-vlp/xd_test_ivlpc.sh caltech101 ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh oxford_pets ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh stanford_cars ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh oxford_flowers ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh food101 ${SEED} ${CFG} ${LOADEP}
done