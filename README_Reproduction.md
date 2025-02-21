
# Reproducibility study of "FairCLIP : Harnessing Fairness in Vision-Language Learning"

> Original paper: (https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_FairCLIP_Harnessing_Fairness_in_Vision-Language_Learning_CVPR_2024_paper.pdf)
> by [Yan Luo*](https://luoyan407.github.io/), [Min Shi*](https://shiminxst.github.io/index.html), Muhammad Osama Khan*, Muhammad Muneeb Afzal, Hao Huang, Shuaihang Yuan,  [Yu Tian](https://yutianyt.com/), Luo Song, Ava Kouhana, [Tobias Elze](http://www.tobias-elze.de/), [Yi Fang](https://engineering.nyu.edu/faculty/yi-fang), and [Mengyu Wang](https://ophai.hms.harvard.edu/team/dr-wang/).


## Abstract

Fairness is a crucial consideration in medical deep learning, as model bias can lead to dis-
parities in diagnoses and treatment decisions. Luo et al. (2024a) conducted a comprehensive
fairness analysis of two vision-language models, CLIP and BLIP2, revealing significant bias
in their predictions. The authors introduced FairCLIP, a model that mitigates bias and
achieves a better performance-fairness trade-off. In this work, we aim to (1) reproduce the
key findings of Luo et al. (2024a) and (2) extend their analysis with additional evaluations.
Our results confirm that most of the reported findings are reproducible, although we identify
discrepancies in specific cases. Furthermore, we conduct a more extensive fairness analysis
by incorporating two additional metrics: Precision Disparity and Mean Absolute Deviation.
Following this analysis, we confirm the presence of bias in CLIP. However, despite being
able to reproduce most of the results, we challenge the claim that FairCLIP improves fair-
ness. Our results suggest that improvements of FairCLIP over CLIP are inconsistent and
architecture- or attribute-dependent, rather than a generalizable improvement in fairness.
Finally, we conduct a study to identify the source of bias. Our results indicate that the
bias does not originate from the summarized clinical notes, attribute identification, medical
pre-training, or group imbalance.

## Installation

The necessary environment is created as follows. 

```bash
conda env create -f fairclip.yml
```

## Dataset
Accessing the dataset can be done upon request. The authors of the original paper stated:

The Harvard-FairVLMed dataset can be accessed via this [link](https://drive.google.com/drive/folders/1bkeifigwOAfnsLvup9mJOSNeA3WsvA2l?usp=drive_link). This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). If you have any questions, please email <harvardophai@gmail.com> and <harvardairobotics@gmail.com>.

Note that, the modifier word “Harvard” only indicates that our dataset is from the Department of Ophthalmology of Harvard Medical School and does not imply an endorsement, sponsorship, or assumption of responsibility by either Harvard University or Harvard Medical School as a legal identity.

The Harvard-FairVLMed dataset comprises 10,000 samples from 10,000 subjects. It is divided into 7,000 training, 1,000 validation, and 2,000 test samples. Upon downloading and extracting these datasets, you will find the dataset structure as follows.

```
Harvard-FairVLMed
├── data_summary.csv
├── gpt-4_summarized_notes.csv
├── Training
├── Validation
└── Test
```
The file split_files.csv details the division of data into training, validation, and testing sets. The data folder contains 10,000 NPZ files named in the format "data_xxxxx.npz", where "xxxxx" (e.g., 06691) is a unique numeric ID. The file meta_all.csv provides metadata (such as race, gender, ethnicity, marital status, age, and preferred language) for each NPZ file. Moreover, the files original_notes.csv and gpt-4_summarized_notes.csv contain original notes and notes summarized by GPT-4, respectively.

Each NPZ file has the following fields.
```
slo_fundus: slo fundus image
md: visual field mean deviation
tds: 52 visual field total deviation values
age: patient age
gender: Female (0), Male (1)
race: Asian (0), Black (1), White (2)
ethnicity: non-Hispanic (0), Hispanic (1), Unknown (-1)
language: English (0), Spanish (1), Other (2), Unknown (-1)
maritalstatus: Marriage or Partnered (0), Single (1), Divorced (2), Widoled (3), Legally Separated (4), Unknown (-1)
glaucoma: Non-Glaucoma (0) or Glaucoma (1)
note: the original de-identified clinical note
note_extra: the original de-identified clinical note with demographic attributes placed at the beginning
```

## Sturcture of code
```
FairCLIP
├──FairCLIP                 Directory with code for fine-tuning models and evaluation with zeroshot
├── mae                     Directory with code for evaluation with linear probing
├── src                     Directory with helper functions
├── LAVIS                   Directory with code for training BLIP. Out of scope for this repoduction study.  
├── moco-v3                 Directory for pre-training CLIP. Use by authors is unknown. Out of scope for this reproduction study.
├──finetune_CLIP.job        Jobfile
├──finetune_FairCLIP.job    Jobfile
├──linprobe_CLIP.job        Jobfile
├──zeroshot_CLIP.job        Jobfile
└──
```
All jobfiles are compatible with a SLURM scheduler. The use of each jobfile is explained below. 

## Pre-training CLIP/FairCLIP
The code for pre-training **CLIP** and **FairCLIP** is in the folder [FairCLIP](./FairCLIP). To run the experiments the following commands should be excecuded. 

For finetuning CLIP run:
```bash
./finetune_CLIP.job
```

For finetuning FairCLIP run:
```bash
./finetune_FairCLIP.job
```

In these jobfiles, you can change the arguments according to the specific experiment, as follows:

For different architectures either 
```bash
DATASET_DIR=./Path/To/Dataset                   # Change to correct path to the dataset
RESULT_DIR=./results                            
SUMMARIZED_NOTE_FILE=gpt4_summarized_notes.csv  # Add the summarized note file
MODEL_ARCH=vit-l14                              # For training different architectures. Options: vit-b16 | vit-l14
MODALITY_TYPE='slo_fundus'
ATTRIBUTE_TYPE=race                             # Attribute which is protected, only relevant for FairCLIP. Options: race | gender | ethnicity | language
TEXT_SOURCE=note                                # For training on clinical note or text corresponding to label. Options: note | text
NUM_EPOCH=10
LR=1e-5
BATCH_SIZE=32
SEED=42 
```

## Evaluation

### Linear Probing
To evalute the CLIP model with linear probing run the following script.

```bash
./linprobe_CLIP.job
```

In this jobfile, you can change the arguments according to the specific experiment, as follows:

```bash
cd FairCLIP/mae
DATA_DIR=./Path/To/Dataset               # Change to correct path to the dataset
FEATS_TYPE=image                         # For specifying the input of linear layer. Options: image | multimodal

PRETRAIN_CHKPT=/Path/to/CKPT             # When evaluating natural CLIP, .... When evaluating a fine-tuned model, provide corresponding path
EXP_NAME=tmp
MODEL_TYPE=clip                                 
```

### Linear Probing
To evalute either the CLIP or the FairCLIP model with zero-shot run the following script.

```bash
./zeroshot_CLIP.job
```

In this jobfiles, you can change the arguments according to the specific experiment, as follows:

``` bash
DATASET_DIR=./Path/To/Dataset                   # Change to correct path to the dataset
RESULT_DIR=./results
MODEL_ARCH=vit-l14                              # For training different architectures. Options: vit-b16 | vit-l14
MODALITY_TYPE='slo_fundus'
LR=1e-5
BATCH_SIZE=32
PREDICT='glaucoma'                              # Prediction of the model. Options: 'glaucoma' | 'race' | 'gender' | 'ethnicity' | 'language'
PRETRAINED_WEIGHTS=./Path/To/Model              # Path to the model to evaluate. If the model is not finetuned: leave empty. If the model is finetuned: change to correct path
SEED=42

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}_${PREDICT}_eval.csv
```

## Acknowledgment and Citation

