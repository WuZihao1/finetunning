"""
集中管理微调任务的所有可配置参数
目的：
1. 避免硬编码，提高代码可维护性
2. 方便进行超参数调优实验
3. 使主要脚本更简洁易读
"""


class TrainingConfig:
    # 模型选择配置
    MODEL_NAME = "bigscience/bloom-560m"  # Hugging Face模型ID或本地路径
    USE_LORA = True  # 是否使用参数高效微调技术LoRA
    QUANTIZE = False  # 是否使用4-bit量化(QLoRA)减少显存占用

    # 数据配置
    DATA_PATH = "data/marketing_sft.json"  # 训练数据路径
    MAX_INPUT_LENGTH = 256  # 输入文本最大长度(超过则截断)
    MAX_OUTPUT_LENGTH = 128  # 输出文本最大长度

    # 训练超参数
    BATCH_SIZE = 4  # 每个GPU的批大小(根据显存调整)
    GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积步数(模拟大batch)
    LEARNING_RATE = 2e-4  # 学习率(典型范围1e-5到2e-4)
    NUM_EPOCHS = 3  # 训练轮数
    WARMUP_RATIO = 0.03  # 学习率预热比例

    # 保存与日志配置
    OUTPUT_DIR = "./results"  # 模型和日志输出目录
    LOGGING_DIR = "./logs"  # Tensorboard日志目录
    LOGGING_STEPS = 10  # 每隔多少步记录日志
    SAVE_STEPS = 200  # 每隔多少步保存模型
    EVAL_STEPS = 200  # 每隔多少步进行评估
    SAVE_TOTAL_LIMIT = 2  # 最多保存的检查点数量

    # 评估配置
    GENERATION_CONFIG = {  # 文本生成参数
        "max_new_tokens": MAX_OUTPUT_LENGTH,
        "temperature": 0.7,  # 0.0(确定性)-1.0+(随机性)
        "top_p": 0.9,  # Nucleus sampling概率阈值
        "do_sample": True,  # 使用采样而非贪心解码
        "repetition_penalty": 1.1,  # >1减少重复
        "num_return_sequences": 1  # 每次生成的样本数
    }