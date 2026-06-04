import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

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

# --- Модель грузим ОДИН раз, до цикла ---
print("Загружаю модель...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Модель загружена.")


def make_note(image_path):
    """Принимает путь к слайду, возвращает текст конспекта."""
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1536)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

# --- Папки: вход slides/, выход notes/ ---
slides_dir = Path("slides") 
notes_dir = Path("notes")
notes_dir.mkdir(exist_ok=True)   # создать notes/, если её нет

slides = sorted(slides_dir.glob("*.png"))   # тут мы сортируем slides по порядку?
print(f"Нашёл слайдов: {len(slides)}")

for slide_path in slides:   # берем слайды по очереди 
    print(f"Обрабатываю {slide_path.name} ...")
    note = make_note(slide_path) # вызываем функцию make_note и передаем в нее слайд 
    out_path = notes_dir / f"{slide_path.stem}.md" # out_path берем имя слайда с приставкой .md. как я понимаю мы создаем слайд в папке notes
    out_path.write_text(f"# Конспект: {slide_path.stem}\n\n{note}", encoding="utf-8") # пишем текст из note в текущий слайд который создали в out_path
    print(f"  -> сохранил {out_path}")

print("Готово. Все конспекты в папке notes/")
