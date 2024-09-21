CFG=vit_b16_p8_c2+2_d3+3_ep50_batch512_small
LOADEP=50

for SEED in 1 2 3
do
    bash scripts/independent-vlp/xd_test_ivlpc.sh fgvc_aircraft ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh dtd ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh sun397 ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh eurosat ${SEED} ${CFG} ${LOADEP}
    bash scripts/independent-vlp/xd_test_ivlpc.sh ucf101 ${SEED} ${CFG} ${LOADEP}
done