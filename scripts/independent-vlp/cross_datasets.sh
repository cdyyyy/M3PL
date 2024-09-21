DEVICE=$1

for SEED in 1 2 3
do
    bash scripts/independent-vlp/xd_test_ivlp.sh caltech101 ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh oxford_pets ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh stanford_cars ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh oxford_flowers ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh food101 ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh fgvc_aircraft ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh dtd ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh sun397 ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh eurosat ${SEED} ${DEVICE}
    bash scripts/independent-vlp/xd_test_ivlp.sh ucf101 ${SEED} ${DEVICE}
done