import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor # тут мы из библиотеки трансформеров импортировали Qwen3VLForConditionalGeneration (я так понимаю это модель Qwen3VL)
#  ForConditionalGeneration тут я не знаю что это значит, AutoProcessor я почитал как я понял это класс в transformers библиотеке, который берет данные которые мы ему даем и превращает их в 
# формат подходящий для нашего Qwen3VL, это может быть например фото, аудио

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct" # тут мы конкретную модель вносим в переменную MODEL_ID это нам нужно для того чтоб код понимал какую конкретно модель скачивать и использовать 

# 1. Загружаем модель и процессор (processor)
print("Загружаю модель (первый раз скачается ~8 ГБ, это нормально, потерпи)...")
model = Qwen3VLForConditionalGeneration.from_pretrained( # в значение model записываем функцию со переменными MODEL_ID то что задали выше это точное название + 4B-Instruct, 
    MODEL_ID,
    dtype=torch.bfloat16,   # тот самый bf16: 2 байта на параметр
    device_map="auto",      # разместить модель на GPU (тут помогает accelerate)
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Модель загружена.")

# 2. Открываем слайд
image = Image.open("first.png")

# 3. Формируем запрос модели
prompt = "Посмотри на этот слайд. Прочитай текст и формулы и объясни простыми словами, о чём он."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

# 4. Превращаем сообщение (картинку + текст) в тензоры, понятные модели
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# 5. Генерируем ответ
print("Думаю над слайдом...")
generated_ids = model.generate(**inputs, max_new_tokens=512)

# 6. Отрезаем токены самого запроса — оставляем только новый ответ модели
trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

# 7. Печатаем
print("\n===== ПОЯСНЕНИЕ =====\n")
print(answer)