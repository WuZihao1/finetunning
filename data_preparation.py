from datasets import load_dataset
from transformers import AutoTokenizer
import config


def load_and_prepare_data():
    """
    加载并预处理监督式微调(SFT)数据
    目的：
    1. 从JSON文件加载原始数据
    2. 将原始数据转换为模型可接受的格式
    3. 应用分词(tokenization)和长度处理
    """
    # 加载JSON格式的数据文件
    dataset = load_dataset(
        'json',
        data_files=config.TrainingConfig.DATA_PATH,
        split='train'
    )

    # 加载与模型匹配的分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.TrainingConfig.MODEL_NAME,
        trust_remote_code=True,
        use_fast=True  # 使用Rust加速的快速分词器
    )

    # 设置分词器的padding token（许多模型没有专门的padding token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        """
        将原始数据转换为模型输入格式
        处理逻辑：
        1. 根据指令(input)和输出(output)构建完整文本序列
        2. 对文本进行分词和截断处理
        3. 生成标签(labels)，仅在output部分计算损失

        技术细节：
        使用-100忽略输入部分的损失计算，只训练输出部分
        """
        prompts = []
        # 遍历每条样本，构建统一的提示格式
        for ins, inp, out in zip(
                examples['instruction'],
                examples.get('input', [''] * len(examples['instruction'])),
                examples['output']
        ):
            # 系统提示设定模型角色和行为
            system_prompt = "你是一个专业的市场营销文案助手。请根据要求生成文案。"

            # 根据是否有附加输入构建完整提示
            if inp:
                prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{ins}\n{inp} [/INST] "
            else:
                prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{ins} [/INST] "
            prompts.append(prompt)

        # 对提示文本进行分词
        model_inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=config.TrainingConfig.MAX_INPUT_LENGTH,
            padding="max_length"  # 填充到统一长度
        )

        # 对输出文本进行分词作为标签
        labels = tokenizer(
            examples["output"],
            truncation=True,
            max_length=config.TrainingConfig.MAX_OUTPUT_LENGTH,
            padding="max_length"
        )["input_ids"]

        # 创建标签并忽略输入部分的损失计算
        # 技术说明：-100在损失函数中会被忽略，确保模型只学习生成输出文本
        ignore_index = -100
        formatted_labels = []
        for label in labels:
            # 将padding位置替换为-100
            formatted_label = [
                ignore_index if token == tokenizer.pad_token_id else token
                for token in label
            ]
            formatted_labels.append(formatted_label)

        model_inputs["labels"] = formatted_labels
        return model_inputs

    # 应用预处理函数到整个数据集
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names  # 移除原始列，保留处理后的特征
    )

    # 分割数据集为训练集和验证集
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    return train_dataset, eval_dataset, tokenizer