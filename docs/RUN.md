# Training and Evaluation

We provide bash scripts in [scripts/](../scripts) for each prompting variant including M3PL, MaPLe, vision, language and independent V-L prompting.
Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory `M3PL/`.
Below we provide training and evaluation instructions for M3PL. The same instructions applies for all other variants including *Vision (VPT), Language and independent V-L prompting* .

## M3PL

#### (1) Base-to-New Generalization Setting
The default training settings are provided in config file at `configs/trainers/M3PL/base2new_*.yaml`. All hyper-parameters such as prompt length, prompt depth can be modified using this config file.

Below, we provide instructions to train M3PL on imagenet. 


```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# seed=1
# trains and evaluates on base classes
bash scripts/m3pl/base2new_train.sh imagenet 1 base2new_imagenet
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh imagenet 1 base2new_imagenet 30

# seed=2
# trains and evaluates on base classes
bash scripts/m3pl/base2new_train.sh imagenet 2 base2new_imagenet
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh imagenet 2 base2new_imagenet 30

# seed=3
# trains and evaluates on base classes
bash scripts/m3pl/base2new_train.sh imagenet 3 base2new_imagenet
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh imagenet 3 base2new_imagenet 30
```

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– M3PL/
|   |   |   |   |   |–– base2new_imagenet/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– M3PL/
|   |   |   |   |   |–– base2new_imagenet/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# prints averaged results for base classes
python parse_test_res.py output/base2new/train_base/imagenet/shots_16/M3PL/base2new_imagenet --multi --n_prompts 8
# averaged results for novel classes
python parse_test_res.py output/base2new/test_new/imagenet/shots_16/M3PL/base2new_imagenet --test-log --multi --n_prompts 8
```

The above steps can be repeated for other individual datasets.


#### (2) Cross-Dataset Generalization Setting
We provide instructions to train M3PL on imageNet using all 1000 classes and then evaluating it directory on new downstream datasets.
We provide cross-dataset config for M3PL: `configs/M3PL/cross_datasets.yaml`.
* Firstly, train M3PL on imagenet in few-shot manner (for all 3 seeds).

```bash
# seed=1 
bash scripts/m3pl/xd_train.sh 1 cross_datasets
# seed=2 
bash scripts/m3pl/xd_train.sh 2 cross_datasets
# seed=3 
bash scripts/m3pl/xd_train.sh 3 cross_datasets
```

* Now evaluate imageNet model on downstream datasets.

```bash
for SEED in 1 2 3
do
    bash scripts/m3pl/xd_test.sh caltech101 ${SEED} cross_datasets 50
    bash scripts/m3pl/xd_test.sh oxford_pets ${SEED} cross_datasets 50
    bash scripts/m3pl/xd_test.sh stanford_cars ${SEED} cross_datasets 50
done
```

#### (3) Domain Generalization 
We use imagenet trained M3PL model for domain generalization experiments. The steps are similar to above cross-dataset experiments, however, model is evaluated on imagenet variants.
* Evaluate imageNet model on variants of imagenet (domain shift datasets).

```bash
for SEED in 1 2 3
do
    bash scripts/m3pl/xd_test.sh imagenetv2 ${SEED} cross_datasets 50
    bash scripts/m3pl/xd_test.sh imagenet_sketch ${SEED} cross_datasets 50
    bash scripts/m3pl/xd_test.sh imagenet_a ${SEED} cross_datasets 50
    bash scripts/m3pl/xd_test.sh imagenet_r ${SEED} cross_datasets 50
done
```


You can obtain averaged results by using the script `parse_test_res.py` and following the similar steps as provided in base-to-novel generalization experiments.
<br>

This should evaluate and save the log files in `output/` directory. To obtain the averaged results, run:

```bash
# prints averaged results for imagenet dataset
python parse_test_res.py output/evaluation/M3PL/cross_datasets/food101 --test-log --multi --n_prompts 8
```


#### Training and Evaluating other variants

For other variants including vision, language and independent V-L prompting techniques, we provide their corresponding configs and scripts as follows.

```
configs
|–– datasets/
|–– trainers/
|   |–– CoCoOp/
|   |–– CoOp/
|   |–– MaPLe/
|   |–– IVLP/
|   |–– VPT/
```

```
scripts
|–– cocoop/
|–– coop/
|–– language-prompting/
|–– maple/
|–– independent-vlp/
```

Please use the corresponding config and script files and follow the same instructions as provided for MaPLe in order to train and evaluate the other variants. Same instructions can be followed to reproduce results of other variants using provided pretrained weights.
This repository also supports using official [CoOp](CoOp.md), [Co-CoOp](Co-CoOp.md) and [MaPLe](MaPLe.md) configs and models.
