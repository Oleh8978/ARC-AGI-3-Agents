# Куди класти і як перевіряти

## 1. Куди пакувати в існуючий репо

Скопіюйте в корінь `ARC-AGI-3-Agents/`:

```
ARC-AGI-3-Agents/
├── world_model/              <- нова папка, скопіювати як є
│   ├── __init__.py            <- створіть порожній файл
│   ├── objects.py
│   └── hypotheses.py
├── tests/
│   ├── test_offline_replay.py
│   └── test_synthetic_ground_truth.py
└── agents/templates/
    └── goal_directed_agent.py   <- сюди інтегрувати (нижче, ще не зроблено)
```

`world_model/` навмисно не залежить від жодного файлу вашого агента — це
чистий модуль, який можна тестувати ізольовано (що ми й зробили).

## 2. Що вже перевірено на вашій машині (тут, в пісочниці)

### `test_synthetic_ground_truth.py` — **4/4 PASSED**
Синтетична гра з відомими наперед правилами (я сам написав ground truth:
гравець рухається на (dx,dy) залежно від дії, стіна блокує рух, монета
зникає при контакті). Движок:
- правильно вивчив дельту руху для кожної з 4 дій без стін,
- **правильно фальсифікував** гіпотезу, коли стіна раз заблокувала рух
  (а не тихо усереднив чи проігнорував протиріччя),
- не переплутав нерухомий об'єкт (ціль) з гравцем,
- виявив правило "монета зникає при сусідстві з гравцем".

Це доводить: **логіка коду коректна**. Запускається без мережі, без API:
```
python tests/test_synthetic_ground_truth.py
```

### `test_offline_replay.py` на ваших реальних `recordings/*.jsonl` — **ЗНАЙШОВ РЕАЛЬНУ ПРОБЛЕМУ, НЕ В МОЄМУ КОДІ**

Прогнав проти всіх 25 записів (goal_directed_agent + hypothesis_agent).
Результат: `action_input.id == 0` (RESET) **у кожному без винятку
записаному кадрі**, у всіх 25 файлах — навіть коли `state == NOT_FINISHED`
сотні кроків підряд.

Я перевірив `choose_action()` в `goal_directed_agent.py` (рядок 350-352):
RESET повертається ТІЛЬКИ коли `state in [NOT_PLAYED, GAME_OVER]`. Тобто
записаний `action_input.id` **не відповідає** реальній логіці вибору дії
— або локальний офлайн-режим генерації записів (без підключення до
сервера) не проброшує справжній id дії у `FrameData.action_input`, або
десь є баг в обгортці `arc_agi`/`arcengine`.

**Це не можна перевірити тут — у мене немає мережі в пісочниці.** Це
треба зробити вам локально:

```bash
# 1. Прогнати короткий live-епізод (5-10 кроків) з реальним підключенням
uv run main.py --agent=goaldirectedagent --game=ls20 --steps=10

# 2. Перевірити свіжий запис
python3 -c "
import json
with open('recordings/<новий_файл>.recording.jsonl') as f:
    for line in f:
        d = json.loads(line)['data']
        print(d.get('state'), d.get('action_input', {}).get('id'))
"
```

Якщо в свіжому записі `action_input.id` теж завжди 0 при `NOT_FINISHED`
— баг реальний, і треба знайти, де саме `FrameData` втрачає це поле
(ймовірно в `arc_agi/local_wrapper.py` чи `remote_wrapper.py` — я бачив
ці файли в архіві, але не встиг проаналізувати саме цей шлях). Якщо
свіжий запис має правильні id — значить проблема тільки в старих
записах (згенеровані іншою версією коду), і можна просто перегенерувати
тестові дані наново.

**Не рухайтесь далі з action-conditioned частиною плану, поки це не
з'ясовано.** До того часу покладайтесь на synthetic-тест як єдине
джерело правди про коректність движка.

## 3. `agents/templates/hypothesis_world_agent.py` — інтеграція готова

Новий агент `HypothesisWorldAgent` перевикористовує ваш робочий
`TransitionGraph`/`GoalDetector`/BFS/`goal_biased_exploration` з
`goal_directed_agent.py` **без змін** (імпортує їх напряму) — міняє
тільки ідентифікацію гравця: замість `ColorRegionTracker` +
хардкодженого `ACTION_DIRECTION` тепер `HypothesisEngine` +
object-centric `extract_objects`. `goal_directed_agent.py` лишається
недоторканим — це ваш baseline для ablation-порівняння.

Запуск на реальній грі (після виправлення багу з `action_input.id`):
```
uv run main.py --agent=hypothesisworldagent --game=ls20
```

### `test_agent_integration_synthetic.py` — прогнав тут, **PASSED**

Це не переписана логіка "начебто так має працювати" — це буквально
імпортований і запущений `HypothesisWorldAgent.choose_action()` /
`.append_frame()`, той самий файл, що піде в прод. Через відсутність
мережі в пісочниці й непрацюючий venv (pydantic не встановлений, власний
`arcengine` конфліктує з numpy) довелось підмінити 4 SDK-символи
(`FrameData`, `GameAction`, `GameState`, `EnvironmentWrapper`)
легковажними dummy-класами — але сам агент виконується без жодних змін.
Прогнав на L-подібному лабіринті зі стіною (класична ситуація, де
Manhattan-евристика Phase 2 провалювалась):

```
reached_goal_at_step: 32
player_identified_at_step: 10
final_player_color: 9   (== справжній колір гравця)
bfs_used: True
```

**Під час цього прогону знайшов і виправив реальний баг** (не
теоретичний): поріг `ImmobileHypothesis.falsified` був `> 1.5`, а
звичайний одноклітинний хід має довжину рівно `1.0` — тобто гравець
ніколи не міг спростувати гіпотезу "я нерухомий", і в summary дійсно
влучив `active_immobile_colors: [1, 3, 9]` (9 = гравець!). Поріг
знижено до `0.5`, тест перезапущено — тепер `active_immobile_colors: [1, 3]`,
гравець правильно виключений. Це саме той тип помилки, який непомітний
на око, але руйнує "Theory"-заявку в статті, якщо його не зловити.

⚠️ **На вашій машині обов'язково перезапустіть цей тест через
`uv run` (з реальним `arcengine`, не заглушками)** — команда та ж:
```
uv run python tests/test_agent_integration_synthetic.py
```
Якщо результат відрізняється від того, що вище — довіряйте своєму
прогону, а не цьому.

## 4. Що ще НЕ зроблено (наступні кроки)

- Інтеграція `world_model/` у `goal_directed_agent.py` (заміна
  `ColorRegionTracker` на `HypothesisEngine`, заміна фіксованого
  `ACTION_DIRECTION` на `predict_delta()`).
- Active action selection (information gain) — модуля ще немає.
- Виправлення багу з `action_input.id` (див. вище) — БЛОКЕР для
  будь-якої подальшої валідації на реальних даних.
