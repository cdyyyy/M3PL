DEVICE=$1

bash scripts/cocoop/base2new.sh imagenet ${DEVICE}
bash scripts/cocoop/base2new.sh caltech101 ${DEVICE}
bash scripts/cocoop/base2new.sh oxford_pets ${DEVICE}
bash scripts/cocoop/base2new.sh stanford_cars ${DEVICE}
bash scripts/cocoop/base2new.sh oxford_flowers ${DEVICE}
bash scripts/cocoop/base2new.sh food101 ${DEVICE}
bash scripts/cocoop/base2new.sh fgvc_aircraft ${DEVICE}
bash scripts/cocoop/base2new.sh dtd ${DEVICE}
bash scripts/cocoop/base2new.sh sun397 ${DEVICE}
bash scripts/cocoop/base2new.sh eurosat ${DEVICE}
bash scripts/cocoop/base2new.sh ucf101 ${DEVICE}