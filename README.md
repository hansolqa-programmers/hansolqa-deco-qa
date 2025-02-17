## 도배 하자 질의 응답 처리 : 한솔데코 시즌2 AI 경진대회

https://dacon.io/competitions/official/236216/overview/description

## 개요

한솔데코는 지속적인 혁신과 기술 개발을 통해 공동 주택 내 실내 마감재 분야에서의 선도적 위치를 더욱 공고히 하고자 합니다. 이와 더불어, 한솔데코는 인공지능(AI) 기술을 이 분야에 접목시켜 혁신을 추진하고 있습니다. AI의 활용은 시트, 마루, 벽면, 도배와 같은 건축의 핵심 자재들의 품질 관리와 하자 판단 과정을 더욱 정교하고 효율적으로 만들어, 이러한 자재들의 관리 및 운용의 질을 향상시키는 데 중점을 두고 있습니다. 이러한 기술적 통합은 고객 만족도를 높이는 동시에, 제품과 서비스의 전반적인 품질 향상에 기여하게 됩니다.

이 대회는 참가자들에게 도배하자와 관련된 다양한 질문과 상황을 제공하고, 이에 대한 정확하고 신속한 응답을 제공하는 AI 모델을 개발하는 것을 목표로 합니다. 이는 실제 현장에서 발생할 수 있는 복잡한 상황에 대응하고, 고객의 문의에 신속하고 정확하게 답변할 수 있는 시스템을 구축하는 데 중요한 역할을 할 것입니다.

## More Info
[teamjjoo_report.pdf](https://github.com/hansolqa-programmers/hansolqa-deco-qa/blob/main/teamjjoo_report.pdf)

## Dependencies

- 해당 코드는 nvidia cuda gpu를 사용할 수 있는 환경에서만 실행 가능합니다.
- nvidia 드라이버 버전과 torch 버전이 맞지 않다는 경고가 뜰 수 있습니다. 이 경우 torch 버전을 올려주세요.

### CASE 1: 리드미의 Demo Code 또는 inference.ipynb을 실행시키려면

```bash
pip install -r requirements_inference.txt
```

### CASE 2: Training Code를 실행시키려면

```bash
pip install -r requirements_train.txt
```

## Quick Inference

- `CASE` 로 모델을 선택해 `COMMON` 으로 Quick Generation

### CASE 1 : SOLAR

```python
base_model_name = 'CurtisJeon/heavytail-kullm-solar-S-4bit'
lora_adaptor_name = 'CurtisJeon/heavytail-kullm-solar-S-lora'
tokenizer_name = "heavytail/kullm-solar-S"
```

### CASE 2: Mistral-7B

```python
base_model_name = 'davidkim205/komt-mistral-7b-v1'
lora_adaptor_name = 'jjookim/beta_1'
tokenizer_name = 'davidkim205/komt-mistral-7b-v1'
```

### Common

```python
# Load Model
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model,
    lora_adaptor_name
)
model.eval()
```

```python
# Load Tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
     trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
```

```python
# Generate
question = "면진 장치가 뭐야?"
reformat_question = f'### 질문: {question}\n### 답변: '
inputs = tokenizer(reformat_question, return_tensors="pt")
inputs = {k:v.cuda() for k, v in inputs.items()}
with torch.no_grad():
    generate_ids = model.generate(
        **inputs, max_new_tokens=300,
    )

generated_answers = tokenizer.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False)[0]
```

## RAG

`models/rag.ipynb` 참고

## 결과

Private: 60위 (상위 11%)
