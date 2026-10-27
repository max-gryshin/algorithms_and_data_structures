# AI Engineering на Go: от скалярного произведения до production-агентов

> Курс построен как совместная работа «профессора» (теория, математика, интуиция — *почему* это работает)
> и «практикующих инженеров» (как это делают в OpenAI/Anthropic/Meta/vLLM — *как* это работает на самом деле,
> где ломается и сколько стоит). Каждая лекция заканчивается работающим Go-кодом, тестами и бенчмарками.

---

## 0. Принципы курса

1. **Build it to understand it.** Каждый ключевой компонент (autograd, attention, BPE, KV-cache, HNSW, агент) пишем сами
   на чистом Go (stdlib). Готовые библиотеки — только для сверки результатов.
2. **Сквозной проект.** Всё складывается в один репозиторий-модель: к середине курса у вас есть **GoGPT** —
   обученный с нуля мини-GPT, а затем загрузчик настоящих весов GPT-2 / Llama-подобной модели, инференс-сервер и
   RAG-агент поверх него.
3. **Теория → формула → форма тензоров → код → тест → эксперимент.** Любая формула записывается вместе с размерностями
   (shape). Если вы не можете назвать shape каждого тензора — вы не поняли формулу.
4. **Gradient check и численные тесты — обязательны.** Каждый backward проверяется конечными разностями.
5. **Ни одной формулы без разбора.** Глубина математики — как в вашем конспекте, но каждая формула разбирается
   «под лупой»: что означает каждый символ, откуда она взялась, и обязательно пересчитывается на маленьких числах
   вручную. Нужный школьный/вузовский минимум (производные, частные производные, суммы, логарифмы, вероятности)
   напоминается прямо в той лекции, где он впервые понадобился, — без отсылок «это вы и так знаете».
6. **Инженерный угол.** В каждой лекции есть блок «Из продакшена»: память, FLOPs, латентность, стоимость, типовые баги.

### Что уже пройдено (по вашему конспекту «Масштабирование в механизме внимания»)

| Тема | Статус | Где углубим |
|---|---|---|
| Токены, Token ID, BPE (на уровне идеи) | ✅ интуиция | L5 — реализуем BPE |
| Матрица эмбеддингов как lookup-таблица | ✅ | L6 — градиенты в эмбеддинги, word2vec-геометрия |
| Позиционные эмбеддинги (синусы, RoPE на уровне идеи) | ✅ интуиция | L6 — реализуем RoPE, доказываем относительность |
| Q, K, V; зачем три проекции; √d_k; mask; softmax | ✅ | L7 — backward, MHA/GQA, тесты |
| Multi-Head, Transformer block, Residual, RMSNorm, FFN | ✅ схема | L8 — собираем и считаем параметры/FLOPs |
| Авторегрессионная генерация | ✅ схема | L10 — сэмплинг, KV-cache |

**Пробелы, которые курс закрывает:** откуда берутся веса (обучение, градиенты, loss), почему это работает
(вероятностная постановка), как это масштабируется (scaling laws, распределённое обучение), как это превращается
в ChatGPT (SFT, RLHF, reasoning), как это обслуживается в проде (инференс, квантизация, батчинг) и как на этом
строят продукты (API, RAG, агенты, evals).

> ⚠️ Замечание к вашему коду `algorythm/attention/attention.go`: функция `softmax` сейчас **не применяет softmax** —
> она делает `MatMul → Scale → MatMul(V)`. Нет exp/нормализации и нет маски. Это первое задание L7 (с тестом,
> который ловит эту ошибку: сумма каждой строки весов внимания должна быть = 1).

---

## 1. Формат каждой лекции

Каждый документ `lectures/LXX_<name>.md` имеет одинаковую структуру:

1. **Зачем это нужно** — какую проблему решает, место в общей картине (карта курса с подсветкой).
2. **Интуиция** — аналогии, рисунки (ASCII-диаграммы), игрушечный пример «на салфетке».
3. **Формальная теория** — определения, формулы с размерностями, выводы (где важно — доказательства).
   Каждая ключевая формула оформлена блоком **🔍 Формула под лупой**: символы → смысл → пересчёт на числах.
   Нужная «забытая» математика — блоками **📐 Математическая справка**.
4. **Числовой пример вручную** — 2–4 числа, которые можно посчитать на бумаге (как ваш пример с «it/works»).
5. **Реализация на Go** — референсный код с комментариями, API пакета.
6. **Из продакшена** — как это сделано в реальных системах, цифры, грабли, ссылки на код (nanoGPT, llama.cpp, vLLM).
7. **Задания** трёх уровней:
   - 🟢 *База* — реализовать/воспроизвести;
   - 🟡 *Инженер* — оптимизировать, обобщить, бенчмарк;
   - 🔴 *Исследователь* — эксперимент/абляция с выводами.
8. **Self-check** — 5–10 вопросов (ответы в конце), «объясни коллеге за 2 минуты».
9. **Литература** — 1 обязательная статья/глава + 2–3 опциональных.

---

## 2. Структура репозитория

```
ai_engineering/
  PLAN.md                    # этот файл
  lectures/                  # L00..L21 *.md
  tensor/                    # L1  — N-мерный тензор, shape/strides, matmul (развитие algorythm/attention/linalg)
  prob/                      # L2  — softmax, logsumexp, cross-entropy, sampling
  autograd/                  # L3–L4 — скалярный, затем тензорный autograd
  optim/                     # L3  — SGD, Momentum, Adam, AdamW, LR schedules
  nn/                        # L4, L6–L8 — Linear, Embedding, RMSNorm, Attention, MLP, Block
  tokenizer/                 # L5  — BPE (byte-level)
  gpt/                       # L8–L10 — модель, обучение, генерация, KV-cache
  weights/                   # L12 — safetensors/GGUF загрузчик, квантизация
  serve/                     # L14 — инференс-сервер, continuous batching, paged KV-cache
  llm/                       # L16 — клиент к LLM API (streaming, retries, cost)
  rag/                       # L17 — BM25, HNSW, chunking, reranking
  agent/                     # L18 — tool loop, MCP-сервер
  evals/                     # L19 — eval-харнесс, LLM-as-judge, статистика
  data/                      # датасеты (tiny shakespeare, русская классика, TinyStories)
```

Стек: Go 1.27, только stdlib (+ `testify` уже есть в go.mod). Опционально `gonum` для сверки линала.
Python — только как «эталон» для сравнения логитов (один скрипт на PyTorch/HF в L12), не обязателен.

---

## 3. Карта курса

```
 Часть I. Фундамент            Часть II. Трансформер           Часть III. Реальные LLM          Часть IV. AI-инженерия
 ─────────────────            ─────────────────────           ───────────────────────          ─────────────────────
 L0 Карта ген. ИИ             L5 Токенизация (BPE)            L11 Scaling laws, распред.        L16 LLM API из Go
 L1 Линал и тензоры    ──►    L6 Эмбеддинги, PE, RoPE   ──►   L12 Реальные веса, квантизация ──► L17 Embeddings и RAG
 L2 Вероятность, CE           L7 Attention: fwd+bwd, MHA      L13 Post-training: SFT/RLHF/DPO   L18 Tool use, агенты, MCP
 L3 Градиенты, autograd       L8 Transformer block            L14 Инференс и serving            L19 Evals, наблюдаемость, safety
 L4 Нейросети, MLP-LM         L9 Обучение GoGPT               L15 Beyond text: диффузия,        L20 Context engineering и
                              L10 Декодинг и KV-cache             VAE, мультимодальность            reasoning-модели
                                                                                               L21 Capstone
```

Ориентир по времени: 1 лекция ≈ 1 неделя (4–6 ч теории + 4–8 ч практики). Весь курс ≈ 5–6 месяцев.
Критический путь: L1→L2→L3→L4→L7→L8→L9→L10. Часть IV можно начинать параллельно после L10.

---

## Часть I. Фундамент: математика, на которой стоит всё

### L0. Карта генеративного ИИ: что такое ChatGPT на самом деле
**Теория:**
- Генеративная модель = модель распределения p(x). LLM = p(следующий токен | контекст). Вся «магия» — это
  предсказание следующего токена, масштабированное до триллионов токенов.
- Жизненный цикл: данные → pretraining (base model) → SFT (instruct) → RLHF/RLAIF/RLVR (assistant, reasoning) →
  инференс → продукт (системный промпт, инструменты, RAG, память, guardrails).
- Почему ChatGPT ≠ GPT: что добавляет каждый этап (base model продолжает текст, assistant отвечает).
- Семейства генеративных моделей: авторегрессионные (GPT), диффузионные (Stable Diffusion), VAE, GAN, flow — где что.
- Архитектуры: encoder-only (BERT), decoder-only (GPT, Llama, Claude), encoder-decoder (T5) — почему победил decoder-only.

**Go:** скелет репозитория, `Makefile` (test/bench/race), утилита `tokcount` — считает «токены» по пробелам и символам
(затравка к L5).

**Задания:** 🟢 Нарисовать (ASCII) полный путь запроса «Привет, как дела?» от клавиатуры до ответа, подписав каждый этап.
🟡 Через любое публичное API получить logprobs и посмотреть топ-5 кандидатов на каждом шаге.
🔴 Эссе на 1 стр.: «Почему предсказание следующего токена приводит к появлению “понимания”?» (compression ↔ prediction).

**Литература:** Karpathy «Intro to LLMs» (видео); Vaswani et al. 2017 «Attention Is All You Need» (обзорно).

---

### L1. Линейная алгебра и тензоры для трансформеров
**Теория:**
- Вектор как точка и как направление; скалярное произведение = |a||b|cos θ → «мера похожести».
- Матричное умножение тремя способами: (1) таблица скалярных произведений строк на столбцы, (2) линейная комбинация
  столбцов, (3) сумма внешних произведений. Почему QKᵀ — это «все пары похожестей сразу».
- Линейное отображение XW как поворот+растяжение+проекция; ранг, низкоранговые матрицы (затравка к LoRA).
- Тензоры: shape, strides, row-major, view vs copy, reshape/transpose без копирования, broadcasting, batch matmul.
- Нормы, косинусное сходство; дисперсия суммы независимых величин (то самое обоснование √d_k — формально).
- Производительность: FLOPs matmul = 2·n·k·p, cache-locality (порядок циклов ikj), блокировка (tiling), SIMD-интуиция,
  memory-bound vs compute-bound (затравка к roofline в L14).

**Go:** пакет `tensor`: `Tensor{data []float32, shape, strides []int}`, `View`, `Reshape`, `Transpose`, `MatMul`,
`BatchMatMul`, broadcasting-`Add/Mul`. Параллельный matmul на горутинах с тайлингом.

**Задания:** 🟢 Перенести `linalg` в `tensor`, покрыть тестами (свойства: (AB)ᵀ = BᵀAᵀ, ассоциативность с допуском).
🟡 Бенчмарк: ijk vs ikj vs tiled vs tiled+goroutines на 512×512 и 2048×2048; посчитать GFLOP/s, сравнить с пиком CPU.
🔴 Эмпирически проверить: Var(q·k) ≈ d_k для случайных N(0,1) векторов при d = 4…1024; построить таблицу и
показать, что после деления на √d_k дисперсия ≈ 1.

**Литература:** 3Blue1Brown «Essence of Linear Algebra»; Goodfellow «Deep Learning», гл. 2.

---

### L2. Вероятность и теория информации: язык, на котором учатся модели
**Теория:**
- Дискретные распределения, условная вероятность, цепное правило: p(x₁…xₙ) = Π p(xᵢ | x<ᵢ) — фундамент LLM.
- Softmax как превращение логитов в распределение; температура; численная стабильность (вычитание max, log-sum-exp).
- Энтропия, cross-entropy, KL-дивергенция; почему минимизация CE = maximum likelihood.
- Perplexity: интерпретация как «среднее число равновероятных вариантов»; bits-per-byte.
- Сэмплирование из категориального распределения (inverse CDF, alias method).
- N-граммные языковые модели, сглаживание; почему они не масштабируются (взрыв контекста) → нужны нейросети.

**Go:** пакет `prob`: `Softmax`, `LogSoftmax`, `LogSumExp`, `CrossEntropy`, `Sample(rng)`, `Perplexity`.
Биграммная модель символов: подсчёт → нормировка → генерация текста.

**Задания:** 🟢 Биграммная модель на «Евгении Онегине» / tiny shakespeare; сгенерировать 500 символов; посчитать
perplexity на валидации. 🟡 Триграммы и сглаживание Лапласа/Kneser-Ney; таблица perplexity n=1..5, где начинается
переобучение. 🔴 Показать на примере, что naive softmax переполняется (логиты ~1000) и что log-sum-exp — нет.

**Литература:** Jurafsky & Martin «Speech and Language Processing», гл. 3 (N-grams); Karpathy «makemore» ч.1.

---

### L3. Градиенты, backprop и оптимизация
**Теория:**
- Производная, градиент, цепное правило; вычислительный граф; forward и backward (reverse-mode autodiff).
- Почему reverse-mode: один скалярный loss, миллиарды параметров → один проход назад.
- Градиенты базовых операций: +, ×, matmul (dA = dC·Bᵀ, dB = Aᵀ·dC), exp, log, softmax+CE (= p − y — красивый вывод).
- Gradient descent → SGD (мини-батчи, шум как регуляризатор) → Momentum → RMSProp → Adam → AdamW (decoupled weight decay).
- Learning rate: warmup, cosine decay; gradient clipping; почему loss «взрывается».
- Градиентная проверка конечными разностями (центральная разность, относительная ошибка).

**Go:** пакет `autograd` (скалярный, как micrograd): `Value{data, grad, children, backward}` + топологическая
сортировка. Пакет `optim`: `SGD`, `Adam`, `AdamW`, `CosineSchedule`.

**Задания:** 🟢 micrograd на Go: обучить линейную регрессию и XOR-сеть. 🟡 Вывести и реализовать градиент
softmax+cross-entropy, проверить gradcheck. 🔴 Сравнить SGD/Momentum/Adam на «функции Розенброка» и на XOR —
траектории, число шагов, чувствительность к LR.

**Литература:** Karpathy «micrograd» (видео); Kingma & Ba 2014 «Adam»; Loshchilov & Hutter 2017 «AdamW».

---

### L4. Нейронные сети: от перцептрона до нейросетевой языковой модели
**Теория:**
- Нейрон, слой Linear (y = xW + b), нелинейности (ReLU, GELU, SiLU) — зачем нелинейность (иначе всё схлопывается
  в одну матрицу). Универсальная аппроксимация (интуитивно).
- Инициализация (Xavier/He): связь с дисперсией из L1; vanishing/exploding gradients.
- Переход от скалярного autograd к тензорному: та же логика, другая гранулярность (производительность ×1000).
- Обучение: train/val/test, overfitting, регуляризация (weight decay, dropout), early stopping, диагностика по
  кривым loss.
- **Нейросетевая LM (Bengio 2003):** эмбеддинги контекста → MLP → softmax по словарю. Первая модель, где эмбеддинги
  *учатся*, а не задаются.

**Go:** тензорный `autograd` поверх `tensor`: операции `MatMul, Add, Tanh, GELU, Embedding, CrossEntropy` с backward.
Пакет `nn`: `Linear`, `Embedding`, `MLP`, `Module` (интерфейс `Parameters()`).

**Задания:** 🟢 MLP-LM по символам (контекст 3–8 символов); val loss лучше биграммной модели из L2.
🟡 Профилирование (`pprof`): где тратится время; оптимизировать аллокации (пулы буферов). 🔴 Абляция
инициализации: обучить с N(0,1), Xavier, He — графики loss и статистики активаций по слоям («мёртвые» нейроны).

**Литература:** Bengio et al. 2003 «A Neural Probabilistic Language Model»; Karpathy «makemore» ч.2–4.

---

## Часть II. Трансформер изнутри

### L5. Токенизация: самый недооценённый источник багов
**Теория:**
- Символы vs слова vs subwords; компромисс «размер словаря ↔ длина последовательности ↔ стоимость».
- BPE: обучение (жадное слияние частых пар), кодирование (применение слияний по рангу), byte-level BPE (UTF-8 байты,
  нет OOV). WordPiece, Unigram/SentencePiece — отличия.
- Специальные токены (BOS/EOS/PAD, chat-токены), pre-tokenization (regex GPT-2/GPT-4).
- Грабли токенизации: кириллица стоит в 2–4 раза дороже по токенам, числа и арифметика, пробелы, «SolidGoldMagikarp»,
  нестабильность при склейке строк, prompt injection через токены.

**Go:** пакет `tokenizer`: `Train(corpus, vocabSize)`, `Encode`, `Decode`, сериализация словаря;
regex-претокенизация; приоритетная очередь для быстрых слияний (ваша тема heap/priority queue!).

**Задания:** 🟢 Обучить BPE (vocab 512/4096) на русском+английском корпусе, `Decode(Encode(s)) == s` для любых
UTF-8 строк (fuzz-тест `go test -fuzz`). 🟡 Ускорить обучение с O(N·merges) до инкрементального обновления пар.
🔴 Сравнить «токенов на слово» для RU/EN/кода при разных размерах словаря; оценить, как это влияет на стоимость API.

**Литература:** Sennrich et al. 2016 (BPE); Karpathy «Let's build the GPT Tokenizer»; tiktoken / minbpe.

---

### L6. Эмбеддинги и позиционное кодирование
**Теория:**
- Embedding = one-hot × матрица; градиент течёт только в использованные строки (sparse update).
- Геометрия эмбеддингов: косинусная близость, аналогии (король − мужчина + женщина), word2vec/skip-gram как
  предшественник; анизотропия эмбеддингов.
- Permutation invariance attention → нужна позиция. Absolute learned, sinusoidal (почему именно синусы: линейная
  выразимость сдвига), **RoPE** (вращение пар координат; доказательство, что ⟨R_m q, R_n k⟩ зависит только от m−n),
  ALiBi. Расширение контекста: NTK-scaling, YaRN, position interpolation.
- Weight tying: матрица эмбеддингов = транспонированный LM-head.

**Go:** `nn.Embedding` с backward, `SinusoidalPE`, `RoPE` (precompute cos/sin таблиц).

**Задания:** 🟢 Реализовать RoPE и тест «относительности»: скалярное произведение после поворота инвариантно к
общему сдвигу позиций. 🟡 Обучить skip-gram на корпусе, найти ближайших соседей для 10 слов. 🔴 Визуализация
(в текст/CSV → график): матрица скалярных произведений PE(i)·PE(j) для синусоидального кодирования.

**Литература:** Su et al. 2021 «RoFormer» (RoPE); Mikolov et al. 2013 (word2vec); Press et al. 2021 (ALiBi).

---

### L7. Attention полностью: forward, backward, Multi-Head, GQA
**Теория (опираясь на ваш конспект, но глубже):**
- Attention как дифференцируемый поиск по словарю (soft lookup); self vs cross attention.
- Пошагово с shape'ами: (B, T, C) → (B, h, T, d_head); QKᵀ: (B,h,T,T); маска; softmax; ·V; concat; W_O.
- Causal mask и padding mask; почему -inf, а не очень маленькое число (и почему на практике берут большое
  отрицательное в fp16).
- **Backward через attention** (вывод градиента softmax-якобиана: dS = A ⊙ (dA − rowsum(dA ⊙ A))).
- Multi-Head: зачем головы (разные подпространства), MQA и **GQA** (Llama 2/3: общие K/V на группу голов →
  KV-cache в разы меньше).
- Сложность O(T²·d) по времени и O(T²) по памяти → мотивация FlashAttention, sliding window, linear attention (L14).
- Интерпретируемость: induction heads, attention sinks.

**Go:** `nn.CausalSelfAttention` (MHA + опционально GQA) с forward/backward; gradcheck-тест.

**Задания:** 🟢 **Исправить `algorythm/attention`**: добавить настоящий построчный stable softmax и causal mask;
тест «строки весов суммируются в 1», «веса будущих токенов = 0»; воспроизвести ваш пример «it/works» (0.19/0.81)
как unit-тест. 🟡 Реализовать batched MHA на `tensor`, сравнить скорость с наивной версией. 🔴 Найти induction
head в обученной в L9 модели: на последовательности вида «A B … A» голова смотрит на токен после предыдущего A.

**Литература:** Vaswani et al. 2017; Ainslie et al. 2023 (GQA); Anthropic «A Mathematical Framework for Transformer
Circuits», «In-context Learning and Induction Heads».

---

### L8. Transformer Block и архитектура GPT целиком
**Теория:**
- Residual stream как «шина данных», в которую слои пишут и из которой читают.
- LayerNorm vs RMSNorm (формулы, backward), Pre-LN vs Post-LN — почему Pre-LN обучается стабильнее.
- FFN: 4×d расширение, GELU; **SwiGLU** (Llama) и почему коэффициент 8/3. FFN как key-value память.
- Сборка GPT-2 vs Llama: таблица отличий (PE, norm, activation, bias, GQA, tying).
- **Подсчёт параметров** (12·L·d² + V·d) и **FLOPs** (≈ 6·N·D на обучение, 2·N на токен инференса);
  память: веса, градиенты, состояния Adam (16 байт/параметр в mixed precision), активации.

**Go:** `nn.RMSNorm`, `nn.LayerNorm`, `nn.SwiGLU`, `nn.Block`, `gpt.Model` (config-driven: GPT-2-style и Llama-style).

**Задания:** 🟢 Собрать модель, проверить форму выхода (B, T, V) и что loss на старте ≈ ln(V) (почему?).
🟡 Калькулятор `gpt/params.go`: по конфигу печатает параметры/память/FLOPs; сверить с GPT-2 124M и Llama-3-8B.
🔴 Абляции на маленькой модели: без residual, Post-LN vs Pre-LN, без масштабирования √d_k — кривые обучения
(экспериментально отвечаем на ваш первый вопрос про √d!).

**Литература:** Radford et al. 2019 (GPT-2); Touvron et al. 2023 (Llama); Xiong et al. 2020 (Pre-LN);
Shazeer 2020 «GLU Variants».

---

### L9. Обучение GoGPT с нуля
**Теория:**
- Пайплайн данных: токенизация корпуса → бинарный файл uint16 → случайные окна длины T → батчи (x, y = x сдвинутый на 1).
- Цикл обучения: forward → loss → backward → clip → optimizer step → schedule; gradient accumulation.
- Методология отладки (рецепт Karpathy): переобучить один батч, проверить loss на init, следить за нормами
  градиентов, update/data ratio, мёртвые нейроны.
- Mixed precision (fp16/bf16) и loss scaling — концептуально; чекпоинты; воспроизводимость (seeds).
- Параллелизм на CPU: data parallel по горутинам с all-reduce градиентов (ваш опыт worker pool / scheduler).

**Go:** `gpt/train.go`: dataloader, тренировочный цикл, логирование (CSV), чекпоинты (encoding/gob или свой бинарный
формат), data-parallel обучение на N горутинах.

**Задания:** 🟢 Обучить ~1–5M параметров на tiny shakespeare / русской классике; val loss < 1.6 (символы) или
сравнимый по BPE; сгенерировать текст. 🟡 Data-parallel по ядрам: ускорение vs число горутин (закон Амдала).
🔴 Мини-scaling-эксперимент: 4 размера модели × 3 объёма данных → построить loss(N, D), увидеть степенной закон.

**Литература:** Karpathy «Let's build GPT from scratch», nanoGPT, llm.c; «A Recipe for Training Neural Networks».

---

### L10. Инференс и декодинг: как модель «говорит»
**Теория:**
- Авторегрессионный цикл; prefill (параллельно по всему промпту, compute-bound) vs decode (по токену, memory-bound).
- **KV-cache:** что кэшируется и почему Q не нужно кэшировать; память = 2·L·h_kv·d_head·T·bytes; влияние GQA.
- Стратегии: greedy, temperature, top-k, top-p (nucleus), min-p, repetition/frequency penalties, beam search
  (почему не используют в чатах).
- **Constrained decoding:** маскирование логитов грамматикой/JSON-schema (как работают structured outputs).
- Speculative decoding: draft-модель + верификация, почему распределение не меняется.
- Stop-условия, streaming токенов, detokenization на границах UTF-8.

**Go:** `gpt/generate.go`: `Generate(ctx, prompt, SamplerConfig) <-chan Token` со streaming через канал,
KV-cache, все сэмплеры; JSON-маска на конечном автомате.

**Задания:** 🟢 KV-cache: доказать тестом, что выход идентичен генерации без кэша; бенчмарк ускорения vs длина.
🟡 Constrained decoding: модель всегда выдаёт валидный JSON заданной схемы. 🔴 Speculative decoding: маленькая
модель как draft к большей; измерить acceptance rate и ускорение.

**Литература:** Holtzman et al. 2019 (nucleus sampling); Leviathan et al. 2022 (speculative decoding);
Willard & Louf 2023 (structured generation).

---

## Часть III. От игрушки к настоящим LLM

### L11. Масштабирование: scaling laws, данные, распределённое обучение
**Теория:**
- Scaling laws (Kaplan 2020, Chinchilla 2022): L(N, D), оптимальное соотношение ~20 токенов на параметр;
  почему сейчас «переобучают» маленькие модели (inference-optimal).
- Данные: Common Crawl → дедупликация (MinHash/LSH), фильтрация качества, смеси доменов, код и математика,
  синтетические данные. Data is the moat.
- Распределённое обучение: Data Parallel, ZeRO/FSDP (шардирование состояний), Tensor Parallel (Megatron),
  Pipeline Parallel, sequence/context parallel; коммуникации (all-reduce, ring).
- Mixture of Experts: роутер top-k, load balancing loss, почему MoE дешевле на токен.
- Emergent abilities и in-context learning — что известно, что спорно.

**Go:** калькулятор бюджета обучения (GPU-часы, $, оптимальные N/D под бюджет); симуляция ring all-reduce на горутинах
и каналах; MoE-слой с top-2 роутингом в `nn`.

**Задания:** 🟢 По Chinchilla: оптимальная модель для 10²¹ FLOPs. 🟡 Ring all-reduce на каналах, сравнить объём
передачи с наивным. 🔴 MinHash-дедупликация корпуса: найти near-duplicates, оценить влияние на val loss GoGPT.

**Литература:** Kaplan et al. 2020; Hoffmann et al. 2022 (Chinchilla); Rajbhandari et al. 2019 (ZeRO);
Fedus et al. 2021 (Switch Transformer); «The Ultra-Scale Playbook» (HF).

---

### L12. Настоящие веса: загрузка, проверка, квантизация
**Теория:**
- Форматы: safetensors (header JSON + сырые тензоры), GGUF (llama.cpp), dtypes (fp32, fp16, bf16 — устройство битов).
- Соответствие имён тензоров архитектуре (как читать config.json модели).
- **Квантизация:** симметричная/асимметричная, per-tensor/per-channel/per-group, int8, int4 (Q4_0/Q4_K), outliers
  (LLM.int8), GPTQ/AWQ — идеи; влияние на perplexity и скорость (memory-bound → меньше байт = быстрее).

**Go:** пакет `weights`: парсер safetensors (mmap через `syscall.Mmap`), конвертация bf16→fp32, загрузка GPT-2 124M
(и/или SmolLM/TinyLlama) в `gpt.Model`; int8/int4 group-квантизация matmul.

**Задания:** 🟢 Загрузить GPT-2 124M, логиты совпадают с эталоном (HF) до 1e-4; сгенерировать текст — **вы запускаете
настоящую LLM в своём Go-коде**. 🟡 Int8-квантизация: размер, скорость tokens/s, Δperplexity. 🔴 Llama-архитектура
(RoPE+RMSNorm+SwiGLU+GQA) на маленькой открытой модели, chat-шаблон, диалог в терминале.

**Литература:** llama.cpp / llama2.c исходники; Dettmers et al. 2022 (LLM.int8); Frantar et al. 2022 (GPTQ);
Lin et al. 2023 (AWQ).

---

### L13. Post-training: как base-модель становится ChatGPT
**Теория:**
- SFT: instruction-данные, chat-шаблоны, маскирование loss на промпте (учим только ответ).
- RLHF: reward model (Bradley–Terry), PPO с KL-штрафом к референсной модели; reward hacking.
- DPO (вывод из RLHF-цели без RL), его варианты; RLAIF и Constitutional AI.
- **Reasoning-модели:** chain-of-thought, RL с проверяемыми наградами (RLVR, GRPO), test-time compute.
- PEFT: **LoRA** (низкоранговое обновление W + BA, связь с рангом из L1), QLoRA.
- Alignment, отказ, sycophancy, галлюцинации — откуда они берутся в процессе обучения.

**Go:** SFT-режим в `gpt/train.go` (loss mask), `nn.LoRA` обёртка над `Linear`, DPO-loss.

**Задания:** 🟢 SFT GoGPT на маленьком наборе «вопрос→ответ» в своём формате шаблона. 🟡 LoRA-дообучение:
обучаем 0.5% параметров, сравниваем с full fine-tune. 🔴 Мини-GRPO: задача «сложи два числа», награда = правильность;
наблюдаем рост accuracy.

**Литература:** Ouyang et al. 2022 (InstructGPT); Rafailov et al. 2023 (DPO); Hu et al. 2021 (LoRA);
Bai et al. 2022 (Constitutional AI); DeepSeek-R1 2025 (GRPO, RLVR).

---

### L14. Эффективный инференс и serving в продакшене
**Теория:**
- Roofline-модель, arithmetic intensity; почему decode упирается в пропускную способность памяти.
- **FlashAttention:** tiling + online softmax (пересчёт максимума на лету), почему не хранится T×T матрица.
- **PagedAttention (vLLM):** KV-cache блоками как виртуальная память; фрагментация.
- **Continuous batching**, prefix caching, chunked prefill, disaggregated prefill/decode.
- Метрики: TTFT, TPOT/ITL, throughput, goodput; SLO; стоимость за миллион токенов.
- Квантизация в serving, tensor parallel на инференсе, speculative decoding в проде.

**Go:** пакет `serve`: online-softmax attention (проверка эквивалентности обычному), аллокатор paged KV-cache,
**планировщик continuous batching** на горутинах (прямое продолжение вашего `problems/concurrency/scheduler`),
HTTP API в стиле OpenAI с SSE-streaming.

**Задания:** 🟢 Online softmax: численно идентичен, память O(T) вместо O(T²). 🟡 Continuous batching vs static batching:
нагрузочный тест (свой генератор нагрузки), графики TTFT/throughput. 🔴 Prefix caching (radix tree по токенам) —
ускорение для общих системных промптов.

**Литература:** Dao et al. 2022/2023 (FlashAttention 1/2); Kwon et al. 2023 (vLLM/PagedAttention);
Yu et al. 2022 (Orca, continuous batching); Zheng et al. 2023 (SGLang, RadixAttention).

---

### L15. За пределами текста: диффузия, VAE, мультимодальность
**Теория:**
- Автоэнкодеры и VAE (ELBO, reparameterization trick); латентное пространство.
- **Диффузионные модели:** прямой процесс зашумления, обратный — предсказание шума; DDPM, score matching интуиция,
  classifier-free guidance; latent diffusion (Stable Diffusion); flow matching.
- ViT: картинка как последовательность патчей-токенов; CLIP: контрастивное обучение текст↔изображение.
- Мультимодальные LLM: image encoder + проектор в пространство эмбеддингов LLM; аудио-токены; генерация видео.

**Go:** DDPM на 2D-данных (спираль/два полумесяца) с MLP-денойзером на нашем autograd; CLIP-style contrastive loss
на игрушечных парах.

**Задания:** 🟢 Обучить 2D-диффузию, сгенерировать точки, вывести CSV и визуализировать. 🟡 Classifier-free guidance
для двух классов точек. 🔴 Сравнить DDPM и flow matching по числу шагов сэмплинга и качеству.

**Литература:** Ho et al. 2020 (DDPM); Lilian Weng «What are Diffusion Models?»; Dosovitskiy et al. 2020 (ViT);
Radford et al. 2021 (CLIP); Lipman et al. 2022 (Flow Matching).

---

## Часть IV. AI-инженерия: строим продукты на LLM

### L16. LLM API из Go: надёжный клиент
**Теория:**
- Messages API: роли, system prompt, multi-turn, параметры (temperature, max_tokens, stop), stop reasons.
- Streaming (SSE), токены и стоимость, контекстное окно, **prompt caching** (почему порядок в промпте важен).
- Надёжность: ретраи с экспоненциальным backoff и jitter, идемпотентность, rate limits (ваш rate limiter!),
  таймауты через `context.Context`, circuit breaker.
- Structured outputs, JSON schema; prompt engineering как инженерия: шаблоны, few-shot, версионирование промптов.
- Выбор модели: качество/латентность/цена, роутинг между моделями.

**Go:** пакет `llm`: провайдер-независимый интерфейс `Client`, реализация для Claude API и для своего `serve`
из L14 (OpenAI-совместимый), SSE-парсер, ретраи, учёт токенов/стоимости, кэширование.

**Задания:** 🟢 CLI-чат со streaming и историей. 🟡 Конкурентный батч-процессор 1000 запросов с rate limit и бюджетом
($ cap). 🔴 Измерить выигрыш prompt caching по латентности и стоимости на длинном системном промпте.

**Литература:** документация Anthropic API (Messages, streaming, prompt caching, structured outputs);
Anthropic «Prompt engineering overview».

---

### L17. Эмбеддинги, векторный поиск и RAG
**Теория:**
- Embedding-модели (bi-encoder), contrastive обучение, нормализация, косинус vs dot.
- Векторный поиск: brute force, IVF, PQ, **HNSW** (иерархический граф малого мира) — сложность, recall/latency trade-off.
- Лексический поиск BM25; гибридный поиск (RRF); reranking cross-encoder'ом.
- RAG-пайплайн: chunking (размер, overlap, по структуре), metadata, query rewriting, contextual retrieval,
  цитирование источников. Когда RAG, когда длинный контекст, когда fine-tune.
- Оценка: recall@k, MRR, nDCG; faithfulness/groundedness ответа.

**Go:** пакет `rag`: BM25-индекс, **HNSW** (ваши графы + heap), chunker для Markdown, RRF-фьюжн, RAG-цепочка поверх `llm`.

**Задания:** 🟢 RAG по **вашему собственному репозиторию лекций** (`problems/**/*.md` + этот курс): «спроси свой конспект».
🟡 HNSW: recall@10 vs ef/M, сравнение с brute force на 100k векторов. 🔴 Golden set из 50 вопросов; сравнить
BM25 / dense / hybrid / hybrid+rerank по recall@5 и качеству ответа.

**Литература:** Malkov & Yashunin 2016 (HNSW); Lewis et al. 2020 (RAG); Anthropic «Contextual Retrieval»;
Robertson & Zaragoza (BM25).

---

### L18. Tool use, агенты и MCP
**Теория:**
- Function calling: как модель «вызывает» инструменты (это просто сгенерированный структурированный текст + цикл в
  вашем коде). Описание инструментов как промпт-инжиниринг.
- Агентный цикл: think → act → observe; критерии остановки; планирование; саб-агенты; workflow vs agent
  (когда хватает детерминированного пайплайна).
- **MCP (Model Context Protocol):** клиент/сервер, tools/resources/prompts, транспорт.
- Память агента: контекст, суммаризация, внешняя память.
- Безопасность: prompt injection (прямой и через данные инструментов), песочница, принцип наименьших привилегий,
  human-in-the-loop для необратимых действий.

**Go:** пакет `agent`: tool registry с JSON-schema из Go-структур (reflect), агентный цикл с лимитами шагов/токенов,
инструменты (поиск по коду, `go test`, калькулятор, RAG из L17); **MCP-сервер на Go**, отдающий ваши алгоритмы как tools.

**Задания:** 🟢 Агент «найди и почини падающий тест» в песочнице (временная директория). 🟡 MCP-сервер, подключённый
к Claude Code. 🔴 Red-team своего агента: 10 prompt-injection атак через содержимое файлов; защиты и метрика.

**Литература:** Anthropic «Building effective agents»; спецификация MCP; Yao et al. 2022 (ReAct);
Greshake et al. 2023 (indirect prompt injection).

---

### L19. Evals, наблюдаемость и безопасность
**Теория:**
- Почему evals — главный инструмент AI-инженера («нельзя улучшить то, что не измерено»).
- Типы: unit-like (детерминированные проверки), golden datasets, LLM-as-judge (смещения: позиция, длина, self-preference),
  pairwise, human eval; бенчмарки (MMLU, HumanEval, SWE-bench) и контаминация.
- Статистика: доверительные интервалы, bootstrap, размер выборки, недетерминизм (несколько прогонов).
- Наблюдаемость: трейсинг LLM-вызовов (OpenTelemetry), стоимость/латентность, логирование промптов, регрессии при
  смене модели/промпта.
- Guardrails, модерация, PII, галлюцинации; оценка безопасности.

**Go:** пакет `evals`: описание кейсов (YAML/JSON), конкурентный раннер, грейдеры (exact, regex, JSON-schema,
LLM-judge), bootstrap CI, отчёт-диф между двумя версиями; OTel-трейсы в `llm`.

**Задания:** 🟢 Eval-набор для RAG из L17, отчёт с CI. 🟡 Регрессионный тест в CI (`go test`-совместимый) на смену
промпта. 🔴 Измерить position bias LLM-судьи: поменять порядок ответов и посчитать долю смены вердикта.

**Литература:** Hamel Husain «Your AI Product Needs Evals»; Zheng et al. 2023 (LLM-as-a-Judge);
Anthropic «Develop tests / evals» docs.

---

### L20. Context engineering и reasoning-модели на практике
**Теория:**
- Контекст как ограниченный ресурс: «context rot», порядок и структура, компакция, суммаризация, передача
  состояния между агентами.
- Reasoning-модели: extended thinking, бюджеты на рассуждения, когда они помогают, а когда — нет; стоимость.
- Длинный контекст: lost-in-the-middle, needle-in-a-haystack; как это связано с attention из L7.
- Паттерны: orchestrator-workers, evaluator-optimizer, параллелизация; мульти-агентные системы.

**Go:** менеджер контекста в `agent` (бюджет токенов, компакция истории, приоритеты), orchestrator-workers на горутинах.

**Задания:** 🟢 Needle-in-a-haystack тест на своём `serve` и на API. 🟡 Компакция истории длинной сессии агента без
потери качества на evals из L19. 🔴 Orchestrator-workers vs один агент на задаче анализа большого репозитория:
качество/стоимость/время.

**Литература:** Anthropic «Effective context engineering for AI agents»; Liu et al. 2023 «Lost in the Middle»;
Wei et al. 2022 (Chain-of-Thought).

---

### L21. Capstone (выбрать один или объединить)
1. **GoGPT end-to-end:** токенизатор → обучение → SFT → квантизация → OpenAI-совместимый сервер с continuous batching
   → eval-отчёт. Всё на чистом Go.
2. **AI-ассистент по знаниям:** RAG + агент + MCP поверх вашей базы алгоритмов и лекций; evals, трейсинг, бюджет.
3. **Мини-инференс-движок:** загрузка открытой Llama-подобной модели, int4, paged KV-cache, prefix caching,
   speculative decoding; бенчмарк против llama.cpp.

Критерии: работающее демо, тесты, бенчмарки, eval-отчёт, README с архитектурной диаграммой и разбором trade-off'ов.

---

## 4. Что дальше

Порядок генерации документов: L0 → L1 → … → L21 (по одной-две лекции за раз, чтобы встраивать ваш фидбек
и проверять код). Каждая лекция генерируется вместе с референсной реализацией пакета и тестами, которые можно
запустить `go test ./ai_engineering/...`.
