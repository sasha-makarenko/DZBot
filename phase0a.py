import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor 
# ForConditionalGeneration это  класс (class) — описание архитектуры, «как модель устроена и под какую задачу подключена»
#  AutoProcessor я почитал как я понял это класс в transformers библиотеке, 
# который берет данные которые мы ему даем и превращает их в формат подходящий для нашего Qwen3VL, это может быть например фото,текст

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct" # идентификатор репозитория на Hugging Face Hub

# 1. Загружаем модель и процессор (processor)
print("Загружаю модель (первый раз скачается ~8 ГБ, это нормально, потерпи)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,   # настройка точности (precision) в torch
    device_map="auto",     
)

processor = AutoProcessor.from_pretrained(MODEL_ID) # тут в processor мы записываем - вызываем AutoProcessor он автоматически загружает нужный инструмент для обработки данных (я это прочитал в интернете не знаю точно то это значит)
# и в AutoProcessor вызываем функцию from_pretrained то есть уже существующую модель в нее передаем переменную MODEL_ID  ВАЖНО точно что значит from_pretrained я не знаю 
print("Модель загружена.")

# 2. Открываем слайд
image = Image.open("first.png")

# 3. Формируем запрос модели
prompt = "Посмотри на этот слайд. Прочитай текст и формулы и объясни простыми словами, о чём он."

messages = [
    {
        "role": "user", # 
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
generated_ids = model.generate(**inputs, max_new_tokens=512) # generated_ids в него передаем наши model и generate с переменными inputs, не знаю что значит **, max_new_tokens=512 задаем кол во токенов 

# 6. Отрезаем токены самого запроса — оставляем только новый ответ модели
trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

# 7. Печатаем
print("\n===== ПОЯСНЕНИЕ =====\n")
print(answer)