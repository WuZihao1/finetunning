from transformers import pipeline, AutoModelForCausalLM
from model_setup import setup_model_and_tokenizer
import torch
import config
from peft import PeftModel


def generate_marketing_copy():
    """
    使用微调后的模型生成营销文案
    目的：
    1. 加载训练好的模型
    2. 接受用户输入/示例输入
    3. 生成符合要求的营销文案
    """
    cfg = config.TrainingConfig

    # 1. 加载模型和分词器
    model, tokenizer = setup_model_and_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 特殊处理：如果使用LoRA且只保存了适配器权重
    if cfg.USE_LORA:
        # 加载基础模型（原始预训练模型）
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        # 将LoRA适配器加载到基础模型上
        model = PeftModel.from_pretrained(
            base_model,
            cfg.OUTPUT_DIR + "/final_model/lora_adapter"
        )
        model = model.to(device)
    else:
        # 直接加载完整模型
        model = model.to(device)

    # 2. 创建文本生成Pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device if device != "cpu" else -1,
    )

    # 3. 提示模板（必须与训练时一致）
    def create_prompt(instruction, input_text=None):
        """构造与训练格式一致的提示"""
        system_prompt = "你是一个专业的市场营销文案助手。请根据要求生成文案。"

        if input_text:
            return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction}\n{input_text} [/INST] "
        return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST] "

    # 4. 示例输入（生产环境中可替换为API接收的输入）
    examples = [
        {
            "instruction": "为情人节促销活动写邮件标题和简短正文",
            "input": "产品：手工定制巧克力礼盒；折扣：第二件半价；限时：2.10-2.14"
        },
        {
            "instruction": "为夏季防晒霜创建微博推广文案",
            "input": "品牌：SunGuard；SPF50+；特点：清爽不粘腻、防水防汗"
        },
        {
            "instruction": "为新款智能手机设计Instagram广告文案",
            "input": "品牌：NovaTech X6；亮点：200MP相机、全天续航、折叠屏"
        }
    ]

    # 5. 生成营销文案
    print("🌟 微调模型营销文案生成演示 🌟\n")
    for i, example in enumerate(examples, 1):
        prompt = create_prompt(example["instruction"], example.get("input"))
        print(f"### 示例 #{i}:")
        print(f"指令: {example['instruction']}")
        if "input" in example:
            print(f"附加信息: {example['input']}")

        # 生成文案
        results = text_generator(
            prompt,
            **cfg.GENERATION_CONFIG
        )

        # 从完整响应中提取生成的文案
        generated_text = results[0]['generated_text']
        # 提取模型生成的部分（从[/INST]标记之后）
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]")[1].strip()

        print("\n生成文案:")
        print(generated_text)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    generate_marketing_copy()