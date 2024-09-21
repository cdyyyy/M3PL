DEVICE=$1

for SEED in 1 2 3
do
    bash scripts/language-prompting/xd_test_lp.sh caltech101 ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh oxford_pets ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh stanford_cars ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh oxford_flowers ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh food101 ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh fgvc_aircraft ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh dtd ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh sun397 ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh eurosat ${SEED} ${DEVICE}
    bash scripts/language-prompting/xd_test_lp.sh ucf101 ${SEED} ${DEVICE}
done