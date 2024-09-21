DEVICE=$1

for SEED in 1 2 3
do
    bash scripts/cocoop/xd_test.sh caltech101 ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh oxford_pets ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh stanford_cars ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh oxford_flowers ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh food101 ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh fgvc_aircraft ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh dtd ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh sun397 ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh eurosat ${SEED} ${DEVICE}
    bash scripts/cocoop/xd_test.sh ucf101 ${SEED} ${DEVICE}
done