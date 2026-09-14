# -*- coding: utf-8 -*-
"""
Exam trainer — тренажёр устных ответов на экзаменационные билеты.

Один прогон:
    выбор билета (по номеру / нечёткий поиск по формулировке / случайный)
        -> ответ голосом на диктофон (любой формат аудио) или текстом
        -> локальное распознавание речи faster-whisper (русский, офлайн)
        -> LLM сверяет ответ с эталоном из файла билетов и даёт разбор
           (оценка N/10, что раскрыто, что упущено, ошибки, совет).

Файл билетов: по умолчанию демо-набор sample_answers.md; если рядом лежит свой
конспект Экзамен_ответы_v2.md — берётся он.

Примеры запуска:
    python exam_trainer.py                 # спросит билет (Enter — случайный), ответ голосом
    python exam_trainer.py 17              # сразу билет №17
    python exam_trainer.py --random        # случайный билет без вопросов
    python exam_trainer.py --text          # ответ текстом, без аудио
    python exam_trainer.py --model base    # точнее распознавание (медленнее)
    python exam_trainer.py --audio ans.m4a # взять готовый аудиофайл

Настройка:
    pip install -r requirements.txt
    Скопируй .env.example -> .env и заполни OPENAI_API_KEY (при необходимости —
    OPENAI_BASE_URL, OPENAI_MODEL, EXAM_AUDIO). См. README.md.
"""
import argparse
import difflib
import os
import random
import re
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv

HERE = Path(__file__).parent


def _locate(name: str) -> Path:
    """Ищет файл рядом со скриптом или на уровень выше (корень проекта)."""
    for base in (HERE, HERE.parent):
        if (base / name).exists():
            return base / name
    return HERE / name


load_dotenv(_locate(".env"))


def _locate_answers() -> Path:
    """Личный конспект (Экзамен_ответы_v2.md, не в репозитории) приоритетнее демо-набора."""
    for name in ("Экзамен_ответы_v2.md", "sample_answers.md"):
        p = _locate(name)
        if p.exists():
            return p
    return _locate("sample_answers.md")


ANSWERS_FILE = _locate_answers()
# `or` вместо значения по умолчанию в getenv: в .env переменная может быть
# объявлена, но пустой (OPENAI_MODEL=) — это тоже должно давать модель по умолчанию.
MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"


# ------------------------- Загрузка билетов -------------------------

def load_tickets(path: Path | None = None) -> dict[int, dict]:
    """Разбирает Экзамен_ответы_v2.md на билеты: номер -> {вопрос, эталон}."""
    text = (path or ANSWERS_FILE).read_text(encoding="utf-8")
    tickets = {}
    # Разбиваем по заголовкам "### Вопрос N. ..."
    parts = re.split(r"^### Вопрос (\d+)\. ", text, flags=re.M)
    # parts: [преамбула, "1", тело1, "2", тело2, ...]
    for i in range(1, len(parts) - 1, 2):
        num = int(parts[i])
        body = parts[i + 1]
        title = body.split("\n", 1)[0].strip()
        tickets[num] = {"вопрос": title, "эталон": body.strip()}
    return tickets


# ------------------------- Выбор билета -------------------------

def normalize(s: str) -> str:
    """Нижний регистр, только буквы/цифры и пробелы — для нечёткого сравнения."""
    return " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())


def find_ticket(query: str, tickets: dict[int, dict]) -> int | None:
    """Находит билет по номеру или по формулировке вопроса (нечётко)."""
    query = query.strip()
    if query.isdigit():
        num = int(query)
        return num if num in tickets else None

    q = normalize(query)
    # Оценка похожести: совпадение слов + посимвольная близость difflib
    scored = []
    for num, t in tickets.items():
        title = normalize(t["вопрос"])
        q_words, t_words = set(q.split()), set(title.split())
        word_score = len(q_words & t_words) / max(len(q_words), 1)
        char_score = difflib.SequenceMatcher(None, q, title).ratio()
        scored.append((word_score + char_score, num))
    scored.sort(reverse=True)

    best_score, best_num = scored[0]
    # Отсекаем мусор: нужно общее слово с названием или высокая близость
    best_words = normalize(tickets[best_num]["вопрос"])
    has_common_word = bool(set(q.split()) & set(best_words.split()))
    if best_score < 0.6 or not has_common_word:
        return None

    print(f"\nПохоже, это билет {best_num}: {tickets[best_num]['вопрос']}")
    ok = input("Он? (Enter — да / n — показать другие варианты): ").strip().lower()
    if ok in ("", "y", "д", "да"):
        return best_num

    print("\nБлижайшие варианты:")
    for _score, num in scored[1:4]:
        print(f"  {num}. {tickets[num]['вопрос']}")
    choice = input("Номер билета (или Enter — отмена): ").strip()
    return int(choice) if choice.isdigit() and int(choice) in tickets else None


def choose_ticket(tickets: dict[int, dict], arg_num: int | None, want_random: bool) -> int:
    """Единая точка выбора: --random / номер из аргумента / интерактивный ввод."""
    if want_random:
        return random.choice(list(tickets))
    if arg_num is not None:
        if arg_num not in tickets:
            sys.exit(f"Билет {arg_num} не найден (есть 1-{max(tickets)}).")
        return arg_num
    # Интерактивно: номер, формулировка или Enter — случайный (имитация экзамена).
    print("Введи билет: номер (1-32), формулировку вопроса или Enter — случайный.")
    query = input("Билет: ").strip()
    if not query:
        return random.choice(list(tickets))
    num = find_ticket(query, tickets)
    if num is None:
        sys.exit("Билет не найден. Проверь номер или формулировку.")
    return num


# ------------------------- Распознавание речи -------------------------

def transcribe(audio_path: str, model_size: str) -> str:
    """Расшифровка аудио в текст локальным faster-whisper (русский)."""
    from faster_whisper import WhisperModel

    print(f"[загружаю модель faster-whisper {model_size}...]")
    try:  # сначала пробуем GPU (NVIDIA/CUDA), при любой проблеме — CPU
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        print("[устройство: GPU (CUDA)]")
    except Exception:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("[устройство: CPU]")
    segments, info = model.transcribe(audio_path, language="ru", vad_filter=True)
    print(f"[распознаю, длительность аудио ~{info.duration:.0f} с — прогресс по фрагментам:]")
    parts = []
    for seg in segments:  # генератор: сегменты приходят по мере распознавания
        parts.append(seg.text)
        pct = min(100, seg.end / info.duration * 100)
        print(f"  [{seg.end:6.0f} с | {pct:3.0f}%] {seg.text.strip()}", flush=True)
    text = " ".join(p.strip() for p in parts).strip()
    print(f"[готово: распознано {len(text)} символов]\n")
    return text


def resolve_audio(arg_audio: str | None) -> str:
    """Путь к аудио: --audio > переменная EXAM_AUDIO (с подтверждением) > ручной ввод."""
    if arg_audio:  # явный аргумент — без вопросов
        p = Path(arg_audio).expanduser()
        if not p.exists():
            sys.exit(f"Файл не найден: {p}")
        return str(p)

    default = os.getenv("EXAM_AUDIO")
    if default:
        ans = input(f"\nФайл {default}? (Enter — да, любой символ — указать другой): ").strip()
        if ans == "":
            p = Path(default).expanduser()
            if not p.exists():
                sys.exit(f"Файл по умолчанию не найден: {p}")
            return str(p)

    audio = input("\nПуть к аудиофайлу (mp3/wav/m4a/ogg): ").strip().strip('"')
    if not Path(audio).exists():
        sys.exit(f"Файл не найден: {audio}")
    return audio


# ------------------------- Ввод ответа текстом -------------------------

def read_text_answer() -> str:
    """Читает многострочный ответ. Завершение: две пустые строки, строка "." или EOF.

    Ctrl+Z в Windows работает только когда он первый символ строки, поэтому даём
    более предсказуемые способы закончить ввод.
    """
    print("\nВводи (или вставь) ответ. Можно в несколько строк и абзацев.")
    print("Закончить ввод: два раза Enter на пустой строке, либо строка с одной точкой.")
    print("-" * 60)

    lines: list[str] = []
    blanks = 0
    while True:
        try:
            line = input()
        except EOFError:  # Ctrl+Z (Windows) / Ctrl+D (*nix) — тоже принимаем
            break
        if line.strip() == ".":
            break
        if line.strip() == "":
            blanks += 1
            if blanks >= 2 and lines:
                break
            lines.append("")
            continue
        blanks = 0
        lines.append(line)

    return "\n".join(lines).strip()


# ------------------------- Оценка ответа -------------------------

SYSTEM = """Ты — доброжелательный, но требовательный экзаменатор. Студент готовится к устному
экзамену и тренируется отвечать на билеты. Тебе дают: формулировку вопроса, эталонный конспект
(части «Тезисно» — ключевые факты, «Примеры» — минимальный код/схемы, «Текстом» — развёрнутый
устный ответ) и расшифровку устного ответа студента (текст получен распознаванием речи, поэтому
не придирайся к опечаткам, пунктуации и оговоркам распознавания — оценивай содержание).

Дай разбор строго в таком формате:

## Оценка: N/10

## Что раскрыто хорошо
- ...

## Что упущено (по пунктам эталона «Тезисно»)
- ... (перечисли конкретные тезисы из эталона, которые студент не назвал)

## Ошибки и неточности
- ... (если фактических ошибок нет — так и напиши)

## Совет
1-2 предложения: на что обратить внимание при повторении этого билета.

Правила: оценивай покрытие тезисов эталона, а не дословность; примеры с другими числами или
переменными считай эквивалентными (5//3 и 10//3 — одно и то же, важен показанный приём, а не
конкретные числа); за структуру и уверенное владение терминами повышай оценку; за фактические
ошибки снижай сильнее, чем за пропуски."""


def grade(question: str, reference: str, student_answer: str) -> None:
    client = openai.OpenAI()  # ключ и base_url берутся из OPENAI_API_KEY / OPENAI_BASE_URL
    user_msg = (
        f"ВОПРОС БИЛЕТА: {question}\n\n"
        f"ЭТАЛОННЫЙ КОНСПЕКТ:\n{reference}\n\n"
        f"ОТВЕТ СТУДЕНТА (расшифровка устной речи):\n{student_answer}"
    )
    print("=" * 60)
    stream = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=16000,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n" + "=" * 60)


# ------------------------- CLI -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Тренажёр устных ответов на экзаменационные билеты")
    ap.add_argument("ticket", nargs="?", type=int,
                    help="номер билета (1-32); без него — программа спросит")
    ap.add_argument("--random", "--случайный", dest="random", action="store_true",
                    help="случайный билет без вопросов")
    ap.add_argument("--text", "--текст", dest="text", action="store_true",
                    help="ввести ответ текстом вместо аудио")
    ap.add_argument("--model", "--модель", dest="model", default="tiny",
                    choices=["tiny", "base", "small", "medium"],
                    help="размер модели Whisper (по умолчанию tiny — быстро даже на слабом CPU; "
                         "base/small/medium точнее, но медленнее)")
    ap.add_argument("--audio", help="путь к готовому аудиофайлу (иначе спросит; см. EXAM_AUDIO)")
    args = ap.parse_args()

    tickets = load_tickets()
    print(f"[билеты: {ANSWERS_FILE.name} — {len(tickets)} шт.]")
    num = choose_ticket(tickets, args.ticket, args.random)
    t = tickets[num]

    print(f"\n{'=' * 60}\nБИЛЕТ {num}. {t['вопрос']}\n{'=' * 60}")

    if args.text:
        answer = read_text_answer()
        print("-" * 60)
        print(f"[принято символов: {len(answer)}]")
    else:
        print("Отвечай вслух на диктофон (телефон/что угодно), сохрани файл.")
        audio = resolve_audio(args.audio)
        answer = transcribe(audio, args.model)
        print(f"РАСШИФРОВКА:\n{answer}\n")

    if len(answer) < 50:
        if args.text:
            sys.exit(f"Принято всего {len(answer)} символов — для разбора нужно минимум 50.\n"
                     f"Похоже, текст не попал в программу. Введи ответ ещё раз и заверши "
                     f"ввод двумя Enter на пустой строке.")
        sys.exit(f"Распознано всего {len(answer)} символов — для разбора нужно минимум 50.\n"
                 f"Проверь, что в аудиофайле есть речь, и попробуй модель точнее: --model base.")

    grade(t["вопрос"], t["эталон"], answer)


if __name__ == "__main__":
    main()
