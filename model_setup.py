from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
import config


def setup_model_and_tokenizer():
    """
    初始化模型和分词器并进行LoRA配置
    目的：
    1. 加载基础语言模型和分词器
    2. 应用4-bit量化(QLoRA)减少显存需求(可选)
    3. 配置LoRA参数高效微调
    4. 打印可训练参数数量

    技术细节：
    - LoRA: 低秩矩阵分解，通过训练少量参数实现高效微调
    - QLoRA: 结合4-bit量化和LoRA，可在消费级GPU微调大模型
    """
    cfg = config.TrainingConfig

    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    tokenizer.padding_side = "right"  # 右侧填充，适配自回归生成

    # 2. 量化配置(QLoRA)
    bnb_config = None
    if cfg.QUANTIZE:
        # 4-bit量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 启用4位量化
            bnb_4bit_quant_type="nf4",  # 特殊数据类型，保持精度
            bnb_4bit_compute_dtype=torch.bfloat16,  # 计算使用bfloat16
            bnb_4bit_use_double_quant=True,  # 嵌套量化，额外节省0.4位/参数
        )

    # 3. 加载基础语言模型
    # 注意：量化配置仅在使用QLoRA时传递
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=bnb_config,  # 量化配置
        device_map="auto",  # 自动分配设备(GPU/CPU)
        torch_dtype=torch.bfloat16 if not cfg.QUANTIZE else None,  # 非量化时使用bfloat16
        use_cache=False  # 训练时不使用kv缓存，节省显存
    )

    # 4. LoRA配置
    if cfg.USE_LORA:
        # 根据模型架构选择目标模块
        # 这些模块将应用LoRA适配器
        target_modules = {
            "bigscience/bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "gpt": ["q_proj", "k_proj", "v_proj", "out_proj"],
            "baichuan": ["W_pack", "o_proj", "gate_proj", "up_proj"],
            "qwen": ["c_attn", "c_proj"],
            "default": ["query_key_value", "dense"]
        }.get(cfg.MODEL_NAME.split('-')[0].lower(), "default")

        # 创建LoRA配置对象
        lora_config = LoraConfig(
            r=8,  # LoRA更新的秩(Rank)，决定参数量
            lora_alpha=32,  # LoRA缩放因子，影响学习率
            target_modules=target_modules,  # 应用LoRA的目标层
            lora_dropout=0.05,  # LoRA层的Dropout率
            bias="none",  # 不训练偏置项
            task_type="CAUSAL_LM"  # 因果语言建模任务
        )

        # 将模型转换为PEFT模型(添加LoRA适配器)
        model = get_peft_model(model, lora_config)

        # 打印可训练参数比例(验证LoRA已正确应用)
        model.print_trainable_parameters()

    return model, tokenizer