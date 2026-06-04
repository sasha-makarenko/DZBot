import json
import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

JUDGE_SYSTEM = """Ты — строгий экзаменатор. Ты видишь слайд лекции и конспект студента по нему.
Оцени, насколько конспект соответствует слайду. Будь придирчив: ищи фактические ошибки,
перепутанные обозначения, выдуманное, неверные формулы и числа.

Критерии (каждый — целое число 1–5, где 5 — идеально):
- faithfulness: конспект не противоречит слайду, обозначения не перепутаны.
- completeness: все ключевые элементы слайда отражены.
- no_hallucination: нет фактов, которых нет на слайде.
- formulas_numbers: формулы и числа переданы верно.

Выведи СТРОГО валидный JSON и больше ничего (без markdown, без обратных кавычек, без текста вне JSON):
{
  "faithfulness": {"score": 0, "comment": ""},
  "completeness": {"score": 0, "comment": ""},
  "no_hallucination": {"score": 0, "comment": ""},
  "formulas_numbers": {"score": 0, "comment": ""},
  "discrepancies": []
}"""

print("Загружаю модель-судью...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Модель загружена.")


def judge(image_path, note_text):
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Вот конспект студента по этому слайду. Оцени.\n\n---\n{note_text}\n---"},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=800)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def parse_json(raw):
    """Достаём JSON, даже если модель обернула его в ``` или добавила текст."""
    start = raw.find("{")
    end = raw.rfind("}")
    return json.loads(raw[start:end + 1])


slides_dir = Path("slides")
notes_dir = Path("notes")

results = []
for slide_path in sorted(slides_dir.glob("*.png")):
    note_path = notes_dir / f"{slide_path.stem}.md"
    if not note_path.exists():
        print(f"Пропускаю {slide_path.name}: нет конспекта")
        continue
    note_text = note_path.read_text(encoding="utf-8")
    print(f"Оцениваю {slide_path.name} ...")
    raw = judge(slide_path, note_text)
    try:
        verdict = parse_json(raw)
    except Exception as e:
        print(f"  !! не смог распарсить JSON ({e}). Сырой ответ:\n{raw[:300]}")
        continue
    results.append((slide_path.stem, verdict))

print("\n===== РЕЗУЛЬТАТЫ EVAL =====\n")
for name, v in results:
    f = v["faithfulness"]["score"]
    c = v["completeness"]["score"]
    h = v["no_hallucination"]["score"]
    fn = v["formulas_numbers"]["score"]
    avg = (f + c + h + fn) / 4
    print(f"{name}: faith={f} compl={c} halluc={h} formulas={fn} | avg={avg:.2f}")
    for d in v.get("discrepancies", []):
        print(f"     - {d}")

Path("eval_results.json").write_text(
    json.dumps([{"slide": n, **v} for n, v in results], ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("\nПодробности сохранены в eval_results.json")