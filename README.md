# RealTalk-CN

This repo contains the code and data of:
[RealTalk-CN: A Realistic Chinese Speech-Text Dialogue Benchmark With Cross-Modal Interaction Analysis]
Our code is developed based on the code in [1].

## Setup
```shell
conda create -n RealTalk python=3.10
conda activate RealTalk
pip install -r requirements.txt
```

## Dataset

The data used in this project is available at [Harvard Dataverse](The dataset link in paper).



## Evaluation
### Step 1: Get the Voice Assistant's Response
To obtain the responses from the voice assistant model, run the following command:
```shell
python main.py --model qwen2 --data single_domain_colloquial --split test --task IC
```

**Supported Arguments:**
- `--model`: Specifies the model to use for generating responses. Replace `qwen2` with the model you want to test (e.g., "glm" "baichuan_audio" "baichuan_omni" "qwen2" "qwen2.5_omini" "minicpm" "naive2" "gpt4o_mini").
- `--data`: Selects the subset of the dataset, which includes ["single_domain_colloquial" "single_domain_system" "multi_domain_colloquial" "multi_domain_system"]
- `--task`: Selects the task of the benchmarking, which includes ["IC", "chat", "multimodality_chat"]
The logs will be saved in the `logs` directory, and the results will be automatically saved in the `results` directory.

### Step2: Automatic GPT-4 Evaluation
Running the `get_GPTScore.ipynb` script will automatically get the GPT score of the prediction file in the result directory, including evaluations with and without reference.


## References
```
[1] Chen et al. (2024). VoiceBench: Benchmarking LLM-Based Voice Assistants. arXiv:2410.17196.
```
