from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from model_setup import setup_model_and_tokenizer
from data_preparation import load_and_prepare_data
import config


def train_model():
    """
    模型微调主训练流程
    目的：
    1. 整合所有组件（模型、数据、训练配置）
    2. 设置训练参数和回调
    3. 执行训练并保存最终模型
    """
    cfg = config.TrainingConfig

    # 1. 初始化模型和分词器
    model, tokenizer = setup_model_and_tokenizer()

    # 2. 加载和预处理数据
    train_dataset, eval_dataset, _ = load_and_prepare_data()

    # 3. 数据整理器 - 动态批处理和填充
    # 专门用于语言建模任务的数据整理
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 使用因果语言建模而不是掩码语言建模
    )

    # 4. 训练参数配置
    training_args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,  # 输出目录
        num_train_epochs=cfg.NUM_EPOCHS,  # 训练轮数
        per_device_train_batch_size=cfg.BATCH_SIZE,  # 每设备批大小
        per_device_eval_batch_size=cfg.BATCH_SIZE * 2,  # 评估批大小可更大
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,  # 梯度累积
        learning_rate=cfg.LEARNING_RATE,  # 学习率
        optim="paged_adamw_8bit" if cfg.QUANTIZE else "adamw_torch",  # 优化器
        logging_dir=cfg.LOGGING_DIR,  # 日志目录
        logging_steps=cfg.LOGGING_STEPS,  # 日志记录频率
        evaluation_strategy="steps" if eval_dataset else "no",  # 评估策略
        eval_steps=cfg.EVAL_STEPS,  # 评估频率
        save_strategy="steps",  # 保存策略
        save_steps=cfg.SAVE_STEPS,  # 保存频率
        save_total_limit=cfg.SAVE_TOTAL_LIMIT,  # 最大保存检查点数
        report_to="tensorboard",  # 报告日志到TensorBoard
        fp16=not cfg.QUANTIZE and torch.cuda.is_available(),  # 混合精度训练
        bf16=cfg.QUANTIZE and torch.cuda.is_available(),  # 量化时用bfloat16
        warmup_ratio=cfg.WARMUP_RATIO,  # 学习率预热
        lr_scheduler_type="cosine",  # 余弦学习率调度
        weight_decay=0.01,  # 权重衰减
        remove_unused_columns=False  # 保留预处理添加的列
    )

    # 5. 创建Trainer对象
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # 6. 训练模型
    print("🚀 开始模型训练...")
    trainer.train()

    # 7. 保存最终模型
    save_path = f"{cfg.OUTPUT_DIR}/final_model"
    trainer.save_model(save_path)

    # 对于LoRA模型，同时保存适配器和基础模型
    if cfg.USE_LORA:
        model.save_pretrained(f"{save_path}/lora_adapter")

    tokenizer.save_pretrained(save_path)
    print(f"✅ 训练完成! 模型已保存至: {save_path}")

    return save_path


if __name__ == "__main__":
    train_model()