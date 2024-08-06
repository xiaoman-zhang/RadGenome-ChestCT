# RadGenome-Chest CT: A Grounded Vision-Language Dataset for Chest CT Analysis


Welcome to the official repository of RadGenome-Chest CT. You can access the dataset via the [HuggingFace repository](https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT)


## Baselines
Before you start, you must install the necessary dependencies. To do so, execute the following commands:

```
conda env create --file environment.yaml  
```

Get dataset from huggingface and save to `./src/RadGenome-ChestCT`.
Download LLaMA3 and save to `./src/Model/Meta-Llama-3-8B-Instruct`.
Get SAY Nano checkpoint from [SAT repository](https://github.com/zhaoziheng/SAT) and save to `./src/Model/`.


### Training 
global baseline: `sbatch global_train_rank8.sh`
grounded baseline : `sbatch grounded_train_rank8.sh`


### Inference
global baseline: `./src/sbatch_script/global_inference`
grounded baseline : `./src/sbatch_script/grounded_attn_inference`


