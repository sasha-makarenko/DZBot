import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# --- НОВОЕ в 0b: правила и структура вынесены в системный промпт ---
SYSTEM_PROMPT = """Ты — преподаватель, который превращает слайд лекции в чёткий учебный конспект.
Прочитай слайд (текст, формулы, диаграммы) и выдай структурированный конспект.

Правила вывода:
- Выводи ТОЛЬКО конспект. Без вступлений, без эмодзи, без лишней болтовни.
- Опирайся СТРОГО на то, что есть на слайде. Ничего не додумывай и не добавляй от себя.
- При определениях и формулах точно сохраняй то, что написано на слайде, не меняй местами обозначения (например, A и B).
- Язык — русский. Технические термины давай на языке оригинала, в скобках — русское пояснение.
- ВСЕ формулы во ВСЕХ разделах оформляй в LaTeX (включая раздел «Пояснение»).
- Строго следуй структуре ниже, с этими же Markdown-заголовками.

Структура:
## Резюме
2–3 предложения: о чём слайд в целом.

## Ключевые понятия
Маркированный список главных идей слайда.

## Термины
Каждый термин с коротким определением (термин — определение).

## Пояснение
Развёрнутое объяснение простыми словами, как репетитор."""

USER_PROMPT = "Вот слайд лекции. Сделай по нему конспект строго по заданной структуре."

print("Загружаю модель...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Модель загружена.")

image = Image.open("first.png")

# --- НОВОЕ в 0b: сообщение из system + user ---
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

print("Думаю над слайдом...")
generated_ids = model.generate(**inputs, max_new_tokens=1536)  # НОВОЕ: подняли лимит, чтобы не обрывалось

trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

print("\n===== КОНСПЕКТ =====\n")

# 0c: сохраняем конспект в .md
slide_name = os.path.splitext("first.png")[0]   # "first.png" -> "first"
out_path = f"{slide_name}_конспект.md"

with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"# Конспект: {slide_name}\n\n")
    f.write(answer)

print(f"Готово. Конспект сохранён в: {out_path}")