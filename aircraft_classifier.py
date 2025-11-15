import requests, io, pandas as pd, csv, time
from PIL import Image
from pathlib import Path
from openai import OpenAI


HF_TOKEN = "..."
REQUEST_DELAY = 3  # задержка чтобы не словить бан

MODELS = {
    "google/vit-base-patch16-224": {
        "type": "vision",
        "url": "https://router.huggingface.co/hf-inference/models/google/vit-base-patch16-224"
    },
    "Qwen/Qwen2.5-VL-72B-Instruct:hyperbolic": {
        "type": "vl",
        "client": None
    }
}

openai_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)
MODELS["Qwen/Qwen2.5-VL-72B-Instruct:hyperbolic"]["client"] = openai_client


def predict_vit(path: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/octet-stream",
    }

    try:
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        r = requests.post(MODELS["google/vit-base-patch16-224"]["url"],
                          headers=headers, data=png_bytes, timeout=30)
        if r.status_code != 200:
            print(f"[VIT] HTTP {r.status_code}: {r.text}")
            return "error"

        preds = r.json()
        if not isinstance(preds, list):
            print(f"[VIT] Bad response: {preds}")
            return "error"

        top_label = preds[0]["label"].lower()
        if any(k in top_label for k in ["warplane", "military", "fighter", "bomber"]):
            return "military"
        elif any(k in top_label for k in ["airliner", "passenger", "civil", "airplane"]):
            return "civilian"
        else:
            return "unknown"
    except Exception as e:
        print(f"[VIT] Error: {e}")
        return "error"


def predict_qwen_vl(url: str) -> str:
    PROMPT = "Civilian or military? Answer only one word."

    try:
        completion = MODELS["Qwen/Qwen2.5-VL-72B-Instruct:hyperbolic"]["client"].chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct:hyperbolic",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }],
            max_tokens=4,
            temperature=0,
        )
        ans = (completion.choices[0].message.content or "").strip().lower()
        pred = "civilian" if "civil" in ans else ("military" if "milit" in ans else "unknown")
        return pred
    except Exception as e:
        print(f"[Qwen] Error: {e}")
        return "error"


def wait_with_countdown(seconds: int, message: str = "Задержка"):
    print(f"\n{message} {seconds} секунды...\n")
    for i in range(seconds, 0, -1):
        time.sleep(1)


def compare_models():
    print("Сравнение моделей \033[32mgoogle/vit-base-patch16-224\033[0m и \033[32mQwen/Qwen2.5-VL-72B-Instruct:hyperbolic\033[0m\n")

    try:
        dataset_vit = pd.read_csv("dataset.csv")  # локальные пути
        dataset_qwen = pd.read_csv("dataset1.csv")  # URL
    except FileNotFoundError as e:
        print(f"Ошибка загрузки датасетов: {e}")
        return

    results = []

    print("-" * 35)
    print("Модель: \033[32mgoogle/vit-base-patch16-224\033[0m")
    print("-" * 35 + "\n")

    vit_results = []
    for i, row in dataset_vit.iterrows():
        img_path = Path(row["image_path"])
        true_label = row["label"]

        if not img_path.exists():
            print(f"Файл не найден: {img_path}")
            pred = "missing"
        else:
            pred = predict_vit(img_path)

        vit_results.append(pred)
        correct = pred == true_label
        results.append({
            "model": "google/vit-base-patch16-224",
            "image": str(img_path),
            "true_label": true_label,
            "pred": pred,
            "correct": correct
        })
        print(f"[\033[32mgoogle/vit-base-patch16-224\033[0m] [{i + 1:02d}] \033[34m{img_path}\033[0m → \033[1m{pred}\033[0m (правильный ответ: {true_label})")

        if i < len(dataset_vit) - 1:
            wait_with_countdown(REQUEST_DELAY)

    print("\n\n" + "-" * 47)
    print("Модель: \033[32mQwen/Qwen2.5-VL-72B-Instruct:hyperbolic\033[0m")
    print("-" * 47 + "\n")

    qwen_results = []
    for i, row in dataset_qwen.iterrows():
        url = row["image_path"]
        true_label = row["label"]

        pred = predict_qwen_vl(url)
        qwen_results.append(pred)
        correct = pred == true_label
        results.append({
            "model": "Qwen/Qwen2.5-VL-72B-Instruct:hyperbolic",
            "image": url,
            "true_label": true_label,
            "pred": pred,
            "correct": correct
        })
        print(f"[\033[32mQwen/Qwen2.5-VL-72B-Instruct:hyperbolic\033[0m] [{i + 1:02d}] {url} → \033[1m{pred}\033[0m (правильный ответ: {true_label})")

        if i < len(dataset_qwen) - 1:
            wait_with_countdown(REQUEST_DELAY)

    print("\n" + "-" * 40)
    print("Результаты")
    print("-" * 40)

    # точность google/vit-base-patch16-224
    vit_correct = sum(1 for r in results if r["model"] == "google/vit-base-patch16-224" and r["correct"])
    vit_total = len(dataset_vit)
    vit_acc = vit_correct / vit_total * 100 if vit_total > 0 else 0

    # точность Qwen/Qwen2.5-VL-72B-Instruct:hyperbolic
    qwen_correct = sum(1 for r in results if r["model"] == "Qwen/Qwen2.5-VL-72B-Instruct:hyperbolic" and r["correct"])
    qwen_total = len(dataset_qwen)
    qwen_acc = qwen_correct / qwen_total * 100 if qwen_total > 0 else 0

    print(f"\033[32mgoogle/vit-base-patch16-224\033[0m:")
    print(f"\033[1mТочность\033[0m: {vit_acc:.2f}% ({vit_correct}/{vit_total})\n")

    print(f"\033[32mQwen/Qwen2.5-VL-72B-Instruct:hyperbolic\033[0m:")
    print(f"\033[1mТочность\033[0m: {qwen_acc:.2f}% ({qwen_correct}/{qwen_total})")

    results_df = pd.DataFrame(results)
    results_df.to_csv("models_comparison_results.csv", index=False)

    return results_df


if __name__ == "__main__":
    compare_models()
