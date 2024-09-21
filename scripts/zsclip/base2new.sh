DEVICE=$1

bash scripts/zsclip/zeroshot_base.sh imagenet vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_base.sh caltech101 vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_base.sh oxford_pets vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_base.sh stanford_cars vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_base.sh oxford_flowers vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_base.sh food101 vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_base.sh fgvc_aircraft vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_base.sh dtd vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_base.sh sun397 vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_base.sh eurosat vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_base.sh ucf101 vit_b16 ${DEVICE}

bash scripts/zsclip/zeroshot_new.sh imagenet vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_new.sh caltech101 vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_new.sh oxford_pets vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_new.sh stanford_cars vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_new.sh oxford_flowers vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_new.sh food101 vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_new.sh fgvc_aircraft vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_new.sh dtd vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_new.sh sun397 vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_new.sh eurosat vit_b16 ${DEVICE}
bash scripts/zsclip/zeroshot_new.sh ucf101 vit_b16 ${DEVICE}

