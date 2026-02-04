<div align="center" style="font-family: charter;">
<h1>MixGRPO:</br>Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE</h1>


<a href="https://arxiv.org/abs/2507.21802" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-MixGRPO-red?logo=arxiv" height="20" /></a>
<a href="https://tulvgengenr.github.io/MixGRPO-Project-Page/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/💻_Project-MixGRPO-blue.svg" height="20" /></a>
<a href="https://huggingface.co/tulvgengenr/MixGRPO" target="_blank">
    <img alt="" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-MixGRPO-ffc107?color=ffc107&logoColor=white" height="20" /></a>

<div>
    <a href="https://scholar.google.com/citations?user=lQsMoJsAAAAJ&hl=en&oi=ao" target="_blank">Junzhe Li</a><sup>1,</sup><sup>2,</sup><sup>3</sup><sup>*</sup>,</span>
    <a href="https://scholar.google.com/citations?user=TSMchWcAAAAJ&hl=en&oi=ao" target="_blank">Yutao Cui</a><sup>1</sup><sup>*</sup>, </span>
    <a href="https://scholar.google.com/citations?hl=en&user=TaM4e4wAAAAJ" target="_blank">Tao Huang</a><sup>1</sup><sup>*</sup>,</span>
    <a href="" target="_blank">Yinping Ma</a><sup>3</sup>,</span>
    <a href="https://scholar.google.com/citations?hl=en&user=0ZZamLoAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Chun Fan</a><sup>3</sup>,</span>
    <a href="" target="_blank">Miles Yang</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?user=igtXP_kAAAAJ&hl=en" target="_blank">Zhao Zhong</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?user=FJwtMf0AAAAJ&hl=zh-CN&oi=ao" target="_blank">Liefeng Bo</a><sup>1</sup></span>
</div>

<div>
    <sup>1</sup>Hunyuan, Tencent&emsp;
    </br>
    <sup>2</sup>School of Computer Science, Peking University&emsp;
    </br>
    <sup>3</sup>Computer Center, Peking University&emsp;
</div>


</div>      

## 📝 News
- [2026/02/03] We supported [Coefficients-Preserving Sampling](https://arxiv.org/abs/2509.05952) (CPS) as a more principled alternative to standard SDE sampling. Comparison tables and enhanced visualization results using CPS have been updated in the [paper](https://arxiv.org/abs/2507.21802) !
- [2025/10/02] We have updated the comparison and visualization results with Flow-GRPO, Flow-DPO (offline), and Flow-DPO (online) on SD3.5-M LoRA in the [paper](https://arxiv.org/abs/2507.21802) !
- [2025/7/30] We released the [model checkpoint](https://huggingface.co/tulvgengenr/MixGRPO) fine-tuned based on [FLUX.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) using the MixGRPO algorithm, with [HPSv2](https://github.com/tgxs002/HPSv2), [ImageReward](https://huggingface.co/zai-org/ImageReward), and [Pick Score](https://github.com/yuvalkirstain/PickScore) as multi-rewards !
- [2025/7/30] We released the [paper](https://arxiv.org/abs/2507.21802) and [code](https://github.com/Tencent-Hunyuan/MixGRPO) !

## 🚀 Quick Start

### Installation

#### 1. Environment setup
```bash
conda create -n MixGRPO python=3.12
conda activate MixGRPO
```

#### 2. Requirements installation
```bash
sudo yum install -y pdsh pssh mesa-libGL # centos
bash env_setup.sh
```
The environment dependency is basically the same as [DanceGRPO](https://github.com/XueZeyue/DanceGRPO).

### Models Preparation

#### 1. FLUX
Download the FLUX HuggingFace repository to `"./data/flux"`.
```bash
mkdir ./data/flux
huggingface-cli login
huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
```

#### 2. Reward Models

##### HPS-v2.1
Download the code of [HPSv2](https://github.com/tgxs002/HPSv2).
```bash
git clone https://github.com/tgxs002/HPSv2.git
```

Download the `"HPS_v2.1_compressed.pt"` and `"open_clip_model.safetensors"` to `"./hps_ckpt"`
```bash
mkdir hps_ckpt
huggingface-cli login
huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./hps_ckpt/
huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./hps_ckpt/
```

##### Pick Score
Run the demo code to automatically download to `"~/.cache/huggingface"`:
```bash
python fastvideo/models/reward_model/pick_score.py \
    --device cuda \
    --http_proxy <Your HTTP_PROXY> \ # Default is None
    --https_proxy <Your HTTPS_PROXY>  # Default is None
```

##### ImageReward
Down the `"ImageReward.pt"` and `"med_config.json"` to `"./image_reward_ckpt"`
```bash
huggingface-cli login
huggingface-cli download --resume-download THUDM/ImageReward med_config.json --local-dir ./image_reward_ckpt/
huggingface-cli download --resume-download THUDM/ImageReward ImageReward.pt --local-dir ./image_reward_ckpt/
```

##### CLIP Score
Run the demo code to automatically download to `"~/.cache/huggingface"`:
```bash
python fastvideo/models/reward_model/clip_score.py \
    --device cuda \
    --http_proxy <Your HTTP_PROXY> \ # Default is None
    --https_proxy <Your HTTPS_PROXY>  # Default is None
```

### Preprocess Data
Adjust the `prompt_path` parameter in `"./scripts/preprocess/preprocess_flux_rl_embeddings.sh"` to obtain the embeddings of the prompt dataset.
```bash
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
```

### Run Training
The training dataset is the training prompts in [HPDv2](https://huggingface.co/datasets/ymhao/HPDv2), as shown in `"./data/prompts.txt"`

We use the `"pdsh"` command for multi-node training with `"torchrun"`. The default resource configuration consists of 4 nodes, each with 8 GPUs, totaling 32 GPUs.

First, set your multi-node IPs in the `data/hosts/hostfile`.

Then, run the following script to set the environment variable `INDEX_CUSTOM` on each node to 0, 1, 2, and 3, respectively.
```bash
bash scripts/preprocess/set_env_multinode.sh
```

Next, set the `wandb_key` to your Weights & Biases (WandB) key in `"./scripts/finetune/finetune_flux_grpo_FastGRPO.sh"`.

Finally, run the following training script:
```bash
bash scripts/finetune/finetune_flux_grpo_FastGRPO.sh
```

### Run Inference
The test dataset is also the test prompts in [HPDv2](https://huggingface.co/datasets/ymhao/HPDv2), as shown in `"./data/prompts_test.txt"`

First, you need to download the MixGRPO model weight `"diffusion_pytorch_model.safetensors"` to the `"./mix_grpo_ckpt"` directory.
```bash
mkdir mix_grpo_ckpt
huggingface-cli login
huggingface-cli download --resume-download tulvgengenr/MixGRPO diffusion_pytorch_model.safetensors --local-dir ./mix_grpo_ckpt/
```
Then, adjust the `Input parameters` in "scripts/inference/inference_flux.sh" (currently set to default) and then execute the single-node script.
```bash
bash scripts/inference/inference_flux.sh
```

### Run Evaluation
Set `prompt_file` to the path of the JSON file generated during inference in "scripts/evaluate/eval_reward.sh". Then run the following single-node script.
```bash
bash scripts/evaluate/eval_reward.sh
```


## 🤝 Acknowledgement

We are deeply grateful for the following GitHub repositories, as their valuable code and efforts have been incredibly helpful:

* [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
* [Flow-GRPO](https://github.com/yifan123/flow_grpo)
* [FastVideo](https://github.com/hao-ai-lab/FastVideo)
* [HPSv2](https://github.com/tgxs002/HPSv2)


## ✏️ Citation

### License
MixGRPO is licensed under the License Terms of MixGRPO. See `./License.txt` for more details.

### Bib
If you find MixGRPO useful for your research and applications, please cite using this BibTeX:
```
@misc{li2025mixgrpounlockingflowbasedgrpo,
      title={MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE}, 
      author={Junzhe Li and Yutao Cui and Tao Huang and Yinping Ma and Chun Fan and Miles Yang and Zhao Zhong},
      year={2025},
      eprint={2507.21802},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.21802}, 
}
```

## 🌟 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Tencent-Hunyuan/MixGRPO&type=Date)](https://www.star-history.com/#Tencent-Hunyuan/MixGRPO&Date)
