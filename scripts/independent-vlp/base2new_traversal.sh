DEVICE=$1

bash scripts/independent-vlp/base2new.sh imagenet ${DEVICE}
bash scripts/independent-vlp/base2new.sh caltech101 ${DEVICE}
bash scripts/independent-vlp/base2new.sh oxford_pets ${DEVICE}
bash scripts/independent-vlp/base2new.sh stanford_cars ${DEVICE}
bash scripts/independent-vlp/base2new.sh oxford_flowers ${DEVICE}
bash scripts/independent-vlp/base2new.sh food101 ${DEVICE}
bash scripts/independent-vlp/base2new.sh fgvc_aircraft ${DEVICE}
bash scripts/independent-vlp/base2new.sh dtd ${DEVICE}
bash scripts/independent-vlp/base2new.sh sun397 ${DEVICE}
bash scripts/independent-vlp/base2new.sh eurosat ${DEVICE}
bash scripts/independent-vlp/base2new.sh ucf101 ${DEVICE}