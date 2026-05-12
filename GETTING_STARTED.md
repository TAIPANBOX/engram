# Getting Started

Покрокова інструкція як запустити Engram з нуля за допомогою Claude Code.

---

## 1. Встанови Claude Code

**macOS / Linux:**
```bash
curl -fsSL https://claude.ai/install.sh | sh
```

**Windows:** через WSL або Git Bash (рідного Windows-інсталера повноцінного немає).

Перевір що працює:
```bash
claude --version
```

---

## 2. Помісти цю папку куди тобі зручно

Наприклад:
```bash
mv engram ~/projects/
cd ~/projects/engram
```

---

## 3. Ініціалізуй git

```bash
git init
git add .
git commit -m "chore: initial scaffold from design"
```

---

## 4. Запусти Claude Code

```bash
claude
```

При першому запуску попросить залогінитись через OAuth або API key.

---

## 5. Перший промпт у сесії

Скопіюй і встав:

```
Прочитай CLAUDE.md і DESIGN.md. Це повний дизайн проєкту Engram —
embeddable cognitive memory layer для AI-агентів.

Не пиши код. Тільки підсумуй коротко що ти зрозумів і запропонуй
конкретний план реалізації v0.1 skeleton (за DESIGN.md розділ 7):
які саме файли треба створити, в якому порядку, і чому.

Після цього я зайду в Plan Mode і ми затвердимо план.
```

---

## 6. Затверди план у Plan Mode

Натисни `Shift+Tab` двічі — Claude Code зайде в Plan Mode (він може читати файли і думати, але **не змінює нічого**).

Обговори план, попроси уточнень, попроси альтернативи. Коли все ок — вийди з Plan Mode (`Shift+Tab` ще раз) і дозволь реалізацію.

---

## 7. Імплементуй v0.1 малими шматками

Не намагайся зробити все за одну сесію. Приблизний порядок:

1. **pyproject.toml + dev tooling** → commit
2. **Схема БД** (`engram/storage.py`) + смоук-тест → commit
3. **`Engram.__init__` + connect**, базовий міграційний код → commit
4. **`observe()` + embedding** → commit
5. **`recall()` (тільки cosine)** → commit
6. **CLI заглушка** (`python -m engram.cli`) → commit

Після кожного кроку: `pytest -x` має пройти, `ruff check` чистий, `mypy engram` без помилок. Тоді commit.

---

## 8. Корисні команди в Claude Code

- `/clear` — очистити контекст між непов'язаними задачами (роби часто, це покращує якість)
- `/init` — згенерувати/оновити CLAUDE.md (у нас він уже є, але якщо знадобиться)
- `/help` — список усіх команд
- `Shift+Tab` × 2 — Plan Mode (план без змін)
- `Esc` — перервати поточну дію

---

## 9. Правила, які зекономлять години

1. **Маленькі сесії, одна задача.** Зробив крок → `/clear` → нова сесія для наступного.
2. **Git-commit після кожного робочого шматка** — твоя страховка.
3. **Тести — не опційно.** Завжди проси Claude *"напиши тест і запусти його"*.
4. **Plan Mode перед будь-чим нетривіальним.** Дешеве задоволення, велика економія.
5. **Якщо щось ламається 2+ рази — зупинись.** Не дай Claude зануритись у спіраль. `/clear` і перепиши промпт з нуля.

---

## 10. Коли v0.1 готовий

- усі тести зелені
- `ruff check`, `mypy engram` чисті
- можеш виконати в Python:
  ```python
  from engram import Engram
  m = Engram(path="./test.engram")
  m.observe("Hello world")
  print(m.recall("hello"))
  ```
- зробити tag: `git tag v0.1.0`

Тоді переходиш до v0.2 (importance + decay) — див. `DESIGN.md` розділ 7.

---

Успіхів! Якщо застрягнеш — повертайся в чат з Claude (не Claude Code), показуй конкретну проблему і питай.
