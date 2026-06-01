import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor # тут мы из библиотеки трансформеров импортировали Qwen3VLForConditionalGeneration (я так понимаю это модель Qwen3VL)
#  ForConditionalGeneration тут я не знаю что это значит
#  AutoProcessor я почитал как я понял это класс в transformers библиотеке, который берет данные которые мы ему даем и превращает их в формат подходящий для нашего Qwen3VL, это может быть например фото, аудио

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct" # тут мы конкретную модель вносим в переменную MODEL_ID это нам нужно для того чтоб код понимал какую конкретно модель скачивать и использовать 

# 1. Загружаем модель и процессор (processor)
print("Загружаю модель (первый раз скачается ~8 ГБ, это нормально, потерпи)...")
model = Qwen3VLForConditionalGeneration.from_pretrained( # мы берем модель из tramsformers  Qwen3VLForConditionalGeneration
    # в значение model записываем функцию from_pretrained с переменными MODEL_ID то что задали выше это точное название + 4B-Instruct, 
    MODEL_ID, # MODEL_ID то что задали выше это точное название + 4B-Instruct, 
    dtype=torch.bfloat16,   # тот самый bf16: 2 байта на параметр # это может варироваться, мы тут как раз задали bfloat о котором мы с тобой говорили 
    device_map="auto",      # разместить модель на GPU (тут помогает accelerate) получается мы тут через accelerate работаем ? 
    # Как это работает ? Модель будет просто скачана на GPU и она будет там всегда? или ее будет код подгружать как то ?
)
# на сколько я понимаю этот блок нужен был для загрузки модели мне на GPU. Это была скачка модели и размещение. dtype и device_map это для accelerate 
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
# я предположу что messages это то что мы передаем в модель. у нас есть "role": "user" и  "content" а content это то что мы передаем, у нас модель сразу знает что мы передадим image и prompt
# то есть мой запрос в Claude например всегда так выглядит ? то есть то что я пишу в строку для своего вопроса в Claude это "type": "text", "text": prompt а  "type": "image", "image": image это если я прикрепряю фото 
# я к тому что это получается messages это стандартный формати того как я обращаюсь к LLM ?

# 4. Превращаем сообщение (картинку + текст) в тензоры, понятные модели
inputs = processor.apply_chat_template(
    messages, 
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
# это функция из библиотеки transformers apply_chat_template
# я думаю лучше ты просто поясни мне эту часть 

# 5. Генерируем ответ
print("Думаю над слайдом...")
generated_ids = model.generate(**inputs, max_new_tokens=512) # generated_ids в него передаем наши model и generate с переменными inputs, не знаю что значит **, max_new_tokens=512 задаем кол во токенов 

# 6. Отрезаем токены самого запроса — оставляем только новый ответ модели
trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
# эти два поясни мне пожалуйста. могу только предположить что batch_decode это мы переводим как бы слова в численном виде в слова в буквенный вид 

# 7. Печатаем
print("\n===== ПОЯСНЕНИЕ =====\n")
print(answer)