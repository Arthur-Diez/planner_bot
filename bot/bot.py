import typing
import random
import os
import re
import asyncio
import spacy
from datetime import datetime, timedelta, timezone
from typing import Optional
from aiogram.utils.deep_linking import create_start_link
from aiogram.filters.command import CommandObject
from aiogram.fsm.storage.base import StorageKey
from openai import AsyncOpenAI

import re, asyncio, json, logging, openai
logging.basicConfig(level=logging.INFO)

from config import OPEN_AI_API_KEY, MODEL
openai_client = openai.AsyncOpenAI(api_key=OPEN_AI_API_KEY)

import dateparser
import asyncpg
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.bot import DefaultBotProperties
from aiogram.filters import CommandStart, StateFilter, Command
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    WebAppInfo
)
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

import math
from contextlib import suppress
from aiogram.filters.callback_data import CallbackData  # опционально

from db_config import DB_CONFIG

CONFLICT_SQL = """
SELECT id, title, start_dt, end_dt, all_day
FROM tasks
WHERE assigned_to_user_id = $1
  AND deleted_at IS NULL
  AND all_day = false
  AND NOT ($3 <= start_dt OR $2 >= COALESCE(end_dt, start_dt + interval '1 minute'))
"""


# Для напоминаний: где хранить «сегодня уже отправляли?»
REMIND_SENT_DDL = """
CREATE TABLE IF NOT EXISTS challenge_reminders_sent (
  id bigserial PRIMARY KEY,
  challenge_id bigint NOT NULL,
  user_id bigint NOT NULL,
  local_day date NOT NULL,
  sent_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (challenge_id, user_id, local_day)
);
"""

async def ensure_aux_tables():
    async with db_pool.acquire() as conn:
        await conn.execute(REMIND_SENT_DDL)

async def fetch_user_tz(uid: int) -> int:
    async with db_pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT tz_offset_min FROM users_planner WHERE telegram_id=$1", uid
        ) or 0

def now_local(offset_min: int) -> datetime:
    return datetime.now(timezone(timedelta(minutes=offset_min)))

# --------------------------------------------------
# Конфигурация и инициализация
# --------------------------------------------------
BOT_TOKEN = os.getenv("TOKEN")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())

db_pool: asyncpg.Pool | None = None

async def init_db() -> None:
    global db_pool
    db_pool = await asyncpg.create_pool(**DB_CONFIG)


@dp.message(Command("webapptest"))
async def open_webapp(message: Message, state: FSMContext):
    if await ensure_friend_name_required(message, state):
        return
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="Открыть WebApp",
            web_app=WebAppInfo(url="https://td-webapp.onrender.com")
        )]
    ])
    await message.answer("Открой мини-приложение через кнопку ниже:", reply_markup=kb)

@dp.message(F.web_app_data)
async def handle_web_app_data(message: Message):
    logging.info(f"💬 WebAppData: {message.web_app_data.data}")
    await message.answer("✅ WebAppData получено!")


planner_inline_kb = InlineKeyboardMarkup(inline_keyboard=[
    [
        InlineKeyboardButton(
            text="📅 Открыть планнер",
            web_app=WebAppInfo(url="https://td-webapp.onrender.com")
        )
    ]
])

@dp.message(Command("planner"))
async def open_planner(message: Message, state: FSMContext):
    if await ensure_friend_name_required(message, state):
        return
    await message.answer(
        "Открой мини-приложение планнера:",
        reply_markup=planner_inline_kb
    )


# --------------------------------------------------
# Клавиатура главного меню
# --------------------------------------------------
main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(
                text="Мой планнер",
                web_app=WebAppInfo(url="https://td-webapp.onrender.com")
            )
        ],
        [
            KeyboardButton(text="Профиль"),
            KeyboardButton(text="Друзья"),
            KeyboardButton(text="Покупки"),
        ],
        [
            KeyboardButton(text="Как работает бот"),
            KeyboardButton(text="Поддержка"),
        ],
    ],
    resize_keyboard=True,
)


# --------------------------------------------------
# FSM для профиля
# --------------------------------------------------
class ProfileStates(StatesGroup):
    full_name = State()
    gender = State()
    birth_date = State()
    tz = State()

class FriendTaskStates(StatesGroup):
    waiting_confirm = State()
    waiting_type = State()


# --------------------------------------------------
# Утилиты для профиля / пользователей
# --------------------------------------------------
async def create_user_if_not_exists(tg_id: int) -> None:
    """Создаёт пустую запись user, если её нет."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users_planner (telegram_id)
            VALUES ($1)
            ON CONFLICT (telegram_id) DO NOTHING
            """,
            tg_id,
        )


async def profile_complete(tg_id: int) -> bool:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT full_name, gender, birth_date, tz_offset_min FROM users_planner WHERE telegram_id=$1",
            tg_id,
        )
    return bool(row and all(row.values()))


# --------------------------------------------------
# Логика обработки времени
__all__ = ("parse_offset_to_minutes", "offset_minutes_to_tz")

_PTN = re.compile(r"^\s*([+-]?)(\d{1,2})(?::?(\d{2}))?\s*$")  # +3, -4:30, +0230

def parse_offset_to_minutes(text: str) -> int | None:
    """
    «+3» → 180, «-4:30» → -270.  Возвращает None при неверном формате.
    Принимаются только значения, кратные 15 минутам.
    """
    m = _PTN.match(text)
    if not m:
        return None
    sign   = -1 if m.group(1) == '-' else 1
    hours  = int(m.group(2))
    mins   = int(m.group(3) or 0)
    if hours > 14 or mins > 59 or mins % 15:
        return None
    return sign * (hours * 60 + mins)

def offset_minutes_to_tz(offset_min: int) -> timezone:
    """timezone() для astimezone()."""
    return timezone(timedelta(minutes=offset_min))

# Хелпер-функция для парса времени
def format_utc_offset(minutes: int) -> str:
    sign = '+' if minutes >= 0 else '-'
    abs_min = abs(minutes)
    return f"{sign}{abs_min // 60}:{abs_min % 60:02}"

# --------------------------------------------------
# Обработчики старт / анкета
# --------------------------------------------------
# --- обработка /start с payload ---
@dp.message(CommandStart(deep_link=True))
async def cmd_start_with_payload(message: Message, command: CommandObject, state: FSMContext):
    tg_id = message.from_user.id
    await create_user_if_not_exists(tg_id)

    # Если пользователь незарегистрирован — сначала анкета
    if not await profile_complete(tg_id):
        await state.set_state(ProfileStates.full_name)
        await message.answer("📝 Привет! Давай заполним короткую анкету. Как тебя зовут?")
        # Сохраняем payload во временные данные
        if command.args and command.args.startswith("friend_"):
            await state.update_data(friend_inviter=command.args)
        return

    # Если есть payload "friend_..." → регистрируем дружбу
    if command.args and command.args.startswith("friend_"):
        inviter_tg_id = int(command.args.split("_", 1)[1])
        async with db_pool.acquire() as conn:
            inviter = await conn.fetchrow(
                "SELECT telegram_id FROM users_planner WHERE telegram_id=$1", inviter_tg_id
            )
            user = await conn.fetchrow(
                "SELECT telegram_id FROM users_planner WHERE telegram_id=$1", tg_id
            )
            if not inviter or not user:
                return

            if inviter["telegram_id"] == user["telegram_id"]:
                await message.answer(
                    "👤 Это ваша личная ссылка. Перешлите её другу, чтобы он добавился сам 😊"
                )
                return

            a, b = sorted([inviter["telegram_id"], user["telegram_id"]])

            existing = await conn.fetchval(
                "SELECT 1 FROM friends WHERE user_a=$1 AND user_b=$2", a, b
            )
            if existing:
                inviter_user = await bot.get_chat(inviter_tg_id)
                inviter_mention = (inviter_user.username and f"@{inviter_user.username}") or inviter_user.full_name or "друг"
                await message.answer(
                    f"👥 Вы уже друзья с {inviter_mention}!\n"
                    "Можешь предложить ему задачу или посмотреть список друзей через кнопку «Друзья»."
                )
                return

            await conn.execute(
                """
                INSERT INTO friends(user_a, user_b, requested_by, status, answered_at)
                VALUES($1, $2, $3, 'accepted', now())
                """,
                a, b, inviter["telegram_id"]
            )

        user_mention = (message.from_user.username and f"@{message.from_user.username}") or message.from_user.full_name or "друга"
        await _start_friend_name_flow(
            inviter_tg_id,
            (a, b),
            intro=f"👥 {user_mention} теперь у тебя в друзьях!",
        )

        inviter_label = await _friend_profile_label(inviter_tg_id)
        await _start_friend_name_flow(
            tg_id,
            (a, b),
            message=message,
            state=state,
            intro=f"🎉 Вы теперь друзья с {inviter_label}!",
        )
    else:
        if await ensure_friend_name_required(message, state):
            return
        await message.answer("👋 Готов работать! Используй кнопки ниже.", reply_markup=main_kb)

@dp.message(Command("start"))
async def cmd_start_fallback(message: Message, state: FSMContext):
    text = message.text or ""
    args = ""
    if " " in text:
        _, args = text.split(" ", 1)

    # создаём объект, аналогичный CommandObject
    class DummyCommand:
        def __init__(self, args): self.args = args
    dummy_cmd = DummyCommand(args)

    # вызываем основной хэндлер
    await cmd_start_with_payload(message, dummy_cmd, state)


@dp.message(ProfileStates.full_name)
async def prof_name(message: Message, state: FSMContext):
    await state.update_data(full_name=message.text.strip())
    await state.set_state(ProfileStates.gender)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Мужской", callback_data="gender_m")],
        [InlineKeyboardButton(text="Женский", callback_data="gender_f")],
    ])
    await message.answer("Укажи пол:", reply_markup=kb)


@dp.callback_query(F.data.startswith("gender_"))
async def prof_gender(callback: CallbackQuery, state: FSMContext):
    gen = callback.data.split("_", 1)[1]
    await state.update_data(gender="м" if gen == "m" else "ж")
    await state.set_state(ProfileStates.birth_date)
    await callback.message.edit_text("Дата рождения (ДД.ММ.ГГГГ):")
    await callback.answer()


@dp.message(ProfileStates.birth_date)
async def prof_birth(message: Message, state: FSMContext):
    try:
        bdate = datetime.strptime(message.text.strip(), "%d.%m.%Y").date()
        await state.update_data(birth_date=bdate)
        await state.set_state(ProfileStates.tz)
        await message.answer("Укажи часовой пояс числом UTC (например 3 или -5):")
    except ValueError:
        await message.answer("Формат неверный. Пример: 12.04.2003")


@dp.message(ProfileStates.tz)
async def prof_tz(message: Message, state: FSMContext):
    offset_min = parse_offset_to_minutes(message.text)
    if offset_min is None:
        await message.answer("⛔️ Укажите UTC-смещение, например <code>-4</code> или <code>+5:30</code>")
        return

    data  = await state.get_data()
    tg_id = message.from_user.id

    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE users_planner
            SET full_name=$1, gender=$2, birth_date=$3, tz_offset_min=$4
            WHERE telegram_id=$5
            """,
            data["full_name"], data["gender"], data["birth_date"],
            offset_min, tg_id
        )
    await state.clear()
    await message.answer("✅ Профиль сохранён!", reply_markup=main_kb)

    # После анкеты — если пользователь пришёл по ссылке друга, создаём дружбу
    inviter_payload = data.get("friend_inviter")
    if inviter_payload and inviter_payload.startswith("friend_"):
        inviter_tg_id = int(inviter_payload.split("_", 1)[1])
        async with db_pool.acquire() as conn:
            inviter = await conn.fetchrow("SELECT telegram_id FROM users_planner WHERE telegram_id=$1", inviter_tg_id)
            user    = await conn.fetchrow("SELECT telegram_id FROM users_planner WHERE telegram_id=$1", tg_id)
            if inviter and user and inviter["telegram_id"] != user["telegram_id"]:
                a, b = sorted([inviter["telegram_id"], user["telegram_id"]])
                existing = await conn.fetchval(
                    "SELECT 1 FROM friends WHERE user_a=$1 AND user_b=$2", a, b
                )
                if not existing:
                    await conn.execute(
                        """
                        INSERT INTO friends(user_a, user_b, requested_by, status, answered_at)
                        VALUES($1, $2, $3, 'accepted', now())
                        """,
                        a, b, inviter["telegram_id"]
                    )

        await _start_friend_name_flow(
            inviter_tg_id,
            (a, b),
            intro=f"👥 {message.from_user.full_name or 'Друг'} теперь у тебя в друзьях!",
        )
        friend_label = await _friend_profile_label(inviter_tg_id)
        await _start_friend_name_flow(
            tg_id,
            (a, b),
            message=message,
            state=state,
            intro=f"🎉 Вы теперь друзья с {friend_label}!",
        )
        return


# --------------------------------------------------
# Добавление задачи
# --------------------------------------------------
# -------- GPT function-calling schema -------------
class TaskStates(StatesGroup):
    waiting_text = State()
    waiting_move_old = State()
    waiting_move_new = State()
    waiting_delete_confirm = State()

# ---------- описание функции для GPT ---------------
GPT_TASK_FUN = {
    "name": "parse_task",
    "description": "Разобрать фразу пользователя на параметры задачи.",
    "parameters": {
        "type": "object",
        "properties": {
            "title":     {"type": "string",  "description": "Короткое название задачи"},
            "start_iso": {"type": "string",  "description": "Дата/время начала в ISO-8601"},
            "end_iso":   {"type": ["string", "null"],
                          "description": "Дата/время конца в ISO-8601, либо null"},
        },
        "required": ["title", "start_iso"]
    },
}


async def gpt_parse(
    text: str,
    tz_offset: int = 3,
    reference_date: Optional[str] = None,
    mode: Optional[str] = None
) -> Optional[dict]:
    """
    Возвращает dict: {title, start_iso, end_iso|None}
    Нормализует случай, когда модель выставила end_iso == start_iso → end_iso=None.
    Также просим модель понимать длительности (“на 30 минут / на 2 часа”).
    """
    try:
        tz = f"UTC{format_utc_offset(tz_offset)}"
        today = datetime.now().strftime("%d.%m.%Y")

        system_prompt = (
            "Ты — парсер естественного языка для планера. "
            "Задача: извлечь название, дату/время начала и (опционально) время конца. "
            "Возвращай результат ЧЕРЕЗ function-call parse_task. "
            "Формат времен: ISO-8601 с часовым поясом пользователя (например 2025-08-25T13:00:00+03:00). "
            "Если указана длительность («на 30 минут», «на 2 часа»), вычисли end_iso = start_iso + длительность. "
            "Если указано только одно время — end_iso не добавляй. "
            "Если указана только дата без времени — ставь время 00:00 (в дальнейшем это трактуется как all-day)."
        )
        if mode == "move_old":
            system_prompt += (
                " В режиме переноса старой задачи особенно важно: если указан только один момент времени, "
                "НЕ добавляй end_iso."
            )

        resp = await openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Текст задачи: «{text}». "
                        f"Часовой пояс пользователя: {tz}. "
                        f"Сегодня: {reference_date or today}. "
                        "Верни строго function-call parse_task с ISO-датами, в часовом поясе пользователя."
                    )
                }
            ],
            tools=[{"type": "function", "function": GPT_TASK_FUN}],
            tool_choice={"type": "function", "function": {"name": "parse_task"}},
            temperature=0,
            max_tokens=150,
        )

        choice = resp.choices[0]
        if hasattr(choice.message, "tool_calls"):
            tool_call = choice.message.tool_calls[0]
            parsed_args = json.loads(tool_call.function.arguments)

            # нормализуем “лишний” end == start
            if parsed_args.get("end_iso") == parsed_args.get("start_iso"):
                parsed_args["end_iso"] = None

            return parsed_args

        return None
    except Exception as err:
        logging.exception("GPT parse error: %s", err)
        return None

# ---------- хэндлеры ----------

@dp.callback_query(F.data == "confirm_task")
async def save_task(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    title = data.get("title")
    if not title:
        await cb.message.edit_text("Ошибка при сохранении задачи.")
        await cb.answer()
        return

    start_dt = datetime.fromisoformat(data["start_iso"])
    end_dt = datetime.fromisoformat(data["end_iso"]) if data["end_iso"] else None

    async with db_pool.acquire() as conn:
        uid = cb.from_user.id  # telegram_id

        # Конфликты
        conflict_tasks = await conn.fetch(
            CONFLICT_SQL, uid, start_dt, end_dt or (start_dt + timedelta(minutes=1))
        )

        if conflict_tasks:
            old = conflict_tasks[0]
            await state.update_data(
                conflict_task_id=old["id"],
                conflict_task_title=old["title"],
                conflict_start=old["start_dt"].isoformat(),
                conflict_end=(old["end_dt"].isoformat() if old["end_dt"] else ""),
                all_day=old.get("all_day", False)
            )

            tz_offset = await conn.fetchval(
                "SELECT tz_offset_min FROM users_planner WHERE telegram_id=$1", uid
            ) or 180

            user_tz = timezone(timedelta(minutes=tz_offset))
            local_start = old['start_dt'].astimezone(user_tz)
            local_end = old['end_dt'].astimezone(user_tz) if old['end_dt'] else None

            old_time_str = local_start.strftime("%d.%m %H:%M")
            if local_end:
                old_time_str += f" — {local_end.strftime('%H:%M')}"

            await cb.message.edit_text(
                f"⚠️ Конфликт: '{old['title']}' в {old_time_str}\nЧто делать?",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🕑 Передвинуть старую", callback_data="move_old")],
                    [InlineKeyboardButton(text="🕓 Передвинуть новую", callback_data="move_new")],
                    [InlineKeyboardButton(text="❌ Удалить старую", callback_data="delete_old")],
                ])
            )
            await cb.answer()
            return

        is_all_day = (not end_dt and start_dt.time() == datetime.min.time())

        await conn.execute(
            """
            INSERT INTO tasks(title, start_dt, end_dt, all_day,
                              assigned_to_user_id, assigned_by_user_id, status)
            VALUES($1,$2,$3,$4,$5,$5,'pending')
            """,
            title, start_dt, end_dt, is_all_day, uid
        )

    await state.clear()
    await cb.message.edit_text("✅ Задача сохранена!")
    await cb.answer()

############# Передвижение старой задачи ##################
# --- move_old ---
@dp.callback_query(F.data == "move_old")
async def move_old(cb: CallbackQuery, state: FSMContext):
    await cb.message.edit_text("⏰ На какое время перенести старую задачу?")
    await state.set_state(TaskStates.waiting_move_old)
    await cb.answer()

@dp.message(TaskStates.waiting_move_old)
async def process_move_old_time(msg: Message, state: FSMContext):
    uid = msg.from_user.id
    async with db_pool.acquire() as conn:
        tz_offset = await conn.fetchval(
            "SELECT tz_offset_min FROM users_planner WHERE telegram_id=$1", uid
        ) or 3

    user_tz = timezone(timedelta(minutes=tz_offset))
    data = await state.get_data()
    new_start = datetime.fromisoformat(data["start_iso"]).replace(tzinfo=user_tz)
    ref_date = new_start.date().strftime("%d.%m.%Y")

    parsed = await gpt_parse(msg.text, tz_offset, reference_date=ref_date, mode="move_old")
    if not parsed:
        await msg.answer("❌ Не удалось разобрать время. Попробуйте ещё раз.")
        return

    # Удаляем end_iso, если GPT добавил его по умолчанию
    if parsed.get("end_iso") == parsed.get("start_iso"):
        parsed["end_iso"] = None

    move_start = datetime.fromisoformat(parsed["start_iso"])
    move_end = datetime.fromisoformat(parsed["end_iso"]) if parsed.get("end_iso") else None
    new_end = (
        datetime.fromisoformat(data["end_iso"]).replace(tzinfo=user_tz)
        if data.get("end_iso")
        else new_start + timedelta(minutes=1)
    )

    # ---- ЛОГИКА ПРОВЕРКИ ПЕРЕСЕЧЕНИЯ ----
    old_is_all_day = data.get("all_day") is True

    if old_is_all_day and new_start.time() == datetime.min.time() and new_end == new_start:
        conflict = False
    else:
        if move_end:
            conflict = not (move_end <= new_start or move_start >= new_end)
        else:
            conflict = new_start <= move_start < new_end

    if conflict:
        move_start_local = move_start.astimezone(user_tz)
        move_range = (
            move_start_local.strftime('%H:%M')
            if not move_end
            else f"{move_start_local.strftime('%H:%M')} — {move_end.astimezone(user_tz).strftime('%H:%M')}"
        )

        new_start_local = new_start.astimezone(user_tz)
        new_end_local = new_end.astimezone(user_tz)
        new_range = (
            new_start_local.strftime('%H:%M')
            if new_start == new_end
            else f"{new_start_local.strftime('%H:%M')} — {new_end_local.strftime('%H:%M')}"
        )

        await msg.answer(
            f"❌ <b>Выбранное время для переноса старой задачи всё ещё пересекается с новой задачей</b>.\n"
            f"<b>Новая задача:</b> {new_range}\n"
            f"<b>Переносимая:</b> {move_range}\n\n"
            f"Пожалуйста, укажите <b>другое время</b>, чтобы задачи не пересекались.",
            parse_mode=ParseMode.HTML
        )
        return

    # Обновляем state
    await state.update_data(
        new_old_start=move_start.isoformat(),
        new_old_end=move_end.isoformat() if move_end else ""
    )

    conflict_start_dt = datetime.fromisoformat(data["conflict_start"]).astimezone(user_tz)
    conflict_time = conflict_start_dt.strftime("%d.%m %H:%M")
    if data.get("conflict_end"):
        conflict_end_dt = datetime.fromisoformat(data["conflict_end"]).astimezone(user_tz)
        conflict_time += f" — {conflict_end_dt.strftime('%H:%M')}"

    # форматируем новую задачу
    new_time = new_start.strftime("%d.%m %H:%M")
    if data.get("end_iso"):
        new_time += f" — {datetime.fromisoformat(data['end_iso']).strftime('%H:%M')}"

    # форматируем ВРЕМЯ переноса старой задачи
    move_start_local = move_start.astimezone(user_tz)
    if move_end:
        move_end_local = move_end.astimezone(user_tz)
        move_time = f"{move_start_local.strftime('%d.%m %H:%M')} — {move_end_local.strftime('%H:%M')}"
    else:
        move_time = move_start_local.strftime('%d.%m %H:%M')

    msg_text = (
        f"✅ Вы перенесли старую задачу «{data['conflict_task_title']}»\n"
        f"{conflict_time} ➔ {move_time}\n\n"
        f"Новая задача «{data['title']}» будет записана на {new_time}.\n"
        "Подтвердить?"
    )

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Подтвердить", callback_data="confirm_move_old")],
        [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")]
    ])
    await msg.answer(msg_text, reply_markup=kb)

@dp.callback_query(F.data == "confirm_move_old")
async def confirm_move_old(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    task_id = data["conflict_task_id"]
    new_old_start = datetime.fromisoformat(data["new_old_start"])
    new_old_end = datetime.fromisoformat(data["new_old_end"]) if data.get("new_old_end") else None
    start_dt = datetime.fromisoformat(data["start_iso"])
    end_dt = datetime.fromisoformat(data["end_iso"]) if data["end_iso"] else None

    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE tasks SET start_dt=$1, end_dt=$2 WHERE id=$3",
            new_old_start, new_old_end, task_id
        )
        uid = cb.from_user.id
        await conn.execute(
            """
            INSERT INTO tasks(title,start_dt,end_dt,
                              assigned_to_user_id,assigned_by_user_id,status)
            VALUES($1,$2,$3,$4,$4,'pending')
            """,
            data["title"], start_dt, end_dt, uid
        )

    await state.clear()
    await cb.message.edit_text("🕑 Задачи успешно обновлены.")
    await cb.answer()


@dp.callback_query(F.data == "confirm_move_new")
async def confirm_move_new(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    start_dt = datetime.fromisoformat(data["start_iso"])
    end_dt = datetime.fromisoformat(data["end_iso"]) if data["end_iso"] else None

    async with db_pool.acquire() as conn:
        uid = cb.from_user.id
        await conn.execute(
            """
            INSERT INTO tasks(title,start_dt,end_dt,
                              assigned_to_user_id,assigned_by_user_id,status)
            VALUES($1,$2,$3,$4,$4,'pending')
            """,
            data["title"], start_dt, end_dt, uid
        )

    await state.clear()
    await cb.message.edit_text("✅ Новая задача успешно добавлена!")
    await cb.answer()



# --------------------------------------------------
# Заглушки остальных кнопок
# --------------------------------------------------
@dp.message(F.text.in_(
    ["Мои планы", "Покупки", "Как работает бот", "Поддержка"]
))
async def placeholders(message: Message, state: FSMContext):
    if await ensure_friend_name_required(message, state):
        return
    await message.answer("Функционал в разработке ✨")


#any_text

############# Передвижение НОВОЙ задачи ##################

@dp.callback_query(F.data == "move_new")
async def move_new(cb: CallbackQuery, state: FSMContext):
    await cb.message.edit_text("⏰ На какое время перенести новую задачу?")
    await state.set_state(TaskStates.waiting_move_new)
    await cb.answer()


@dp.message(TaskStates.waiting_move_new)
async def process_move_new_time(msg: Message, state: FSMContext):
    try:
        uid = msg.from_user.id
        async with db_pool.acquire() as conn:
            tz_offset = await conn.fetchval(
                "SELECT tz_offset_min FROM users_planner WHERE telegram_id=$1", uid
            ) or 3
        user_tz = timezone(timedelta(minutes=tz_offset))

        data = await state.get_data()
        old_start = datetime.fromisoformat(data["conflict_start"])
        old_end   = datetime.fromisoformat(data["conflict_end"]) if data.get("conflict_end") else old_start

        # дата той «новой» задачи, которую мы сейчас переносим
        ref_date   = datetime.fromisoformat(data["start_iso"]).astimezone(user_tz).date()
        ref_date_s = ref_date.strftime("%d.%m.%Y")

        # --- GPT ---
        parsed = await gpt_parse(msg.text, tz_offset, reference_date=ref_date_s)
        if not parsed:
            await msg.answer("❌ Не удалось разобрать время. Попробуйте ещё раз.")
            return

        # если GPT склеил start == end → считаем, что конец отсутствует
        if parsed.get("end_iso") == parsed.get("start_iso"):
            parsed["end_iso"] = None

        parsed_start = datetime.fromisoformat(parsed["start_iso"])
        parsed_end   = datetime.fromisoformat(parsed["end_iso"]) if parsed.get("end_iso") else parsed_start
        parsed_date  = parsed_start.date()

        # --- главное правило ---
        # если юзер НЕ указал дату, GPT вернёт «сегодня»; тогда подставляем ref_date
        if parsed_date == datetime.now(user_tz).date() and parsed_date != ref_date:
            # берём только время
            new_start = datetime.combine(ref_date, parsed_start.timetz()).replace(tzinfo=user_tz)
            new_end   = datetime.combine(ref_date, parsed_end.timetz()).replace(tzinfo=user_tz) \
                        if parsed.get("end_iso") else new_start
        else:
            new_start, new_end = parsed_start, parsed_end

        # --- проверка пересечения со старой задачей ---
        conflict = not (new_end   <= old_start or
                        new_start >= old_end)

        if conflict:
            await msg.answer(
                "❌ <b>Новая задача всё ещё пересекается со старой</b>.\n"
                f"<b>Старая:</b> {old_start.astimezone(user_tz):%H:%M} — {old_end.astimezone(user_tz):%H:%M}\n"
                f"<b>Новая:</b> {new_start:%H:%M} — {new_end:%H:%M}",
                parse_mode=ParseMode.HTML,
            )
            return

        # --- сохраняем во FSM и спрашиваем подтверждение ---
        await state.update_data(
            start_iso=new_start.isoformat(),
            end_iso=new_end.isoformat() if new_end != new_start else ""
        )

        time_str = (f"{new_start:%d.%m %H:%M}"
                    f"{' — ' + new_end.strftime('%H:%M') if new_end != new_start else ''}")

        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="✅ Подтвердить", callback_data="confirm_move_new")],
            [InlineKeyboardButton(text="❌ Отмена",       callback_data="cancel")]
        ])
        await msg.answer(f"✅ Новая задача будет перенесена на {time_str}.\nПодтверждаешь?", reply_markup=kb)

    except Exception as e:
        # покажем ошибку в логи и пользователю (чтобы не «молчал»)
        logging.exception("move_new error")
        await msg.answer("🚫 Произошла ошибка при разборе времени. Проверь формат и попробуй ещё раз.")



############# Удаление старой задачи ##################

@dp.callback_query(F.data == "delete_old")
async def delete_old(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    task_id = data["conflict_task_id"]
    start_dt = datetime.fromisoformat(data["start_iso"])
    end_dt = datetime.fromisoformat(data["end_iso"]) if data["end_iso"] else None

    async with db_pool.acquire() as conn:
        # жесткое удаление
        await conn.execute("DELETE FROM tasks WHERE id=$1", task_id)
        # мягкое удаление (альтернатива):
        # await conn.execute("UPDATE tasks SET deleted_at=now() WHERE id=$1", task_id)

        uid = cb.from_user.id
        await conn.execute(
            """
            INSERT INTO tasks(title,start_dt,end_dt,
                              assigned_to_user_id,assigned_by_user_id,status)
            VALUES($1,$2,$3,$4,$4,'pending')
            """,
            data["title"], start_dt, end_dt, uid
        )

    await state.clear()
    await cb.message.edit_text("✅ Старая задача удалена, новая добавлена вместо неё.")
    await cb.answer()



########################################################################################################################
########### НОВЫЙ ФУНКЦИОНАЛ ДОБАВЛЕНИЯ В ДРУЗЬЯ #######################################################################
class FriendStates(StatesGroup):
    awaiting_friend_name = State()

async def _ensure_bot_id() -> int:
    """Возвращает идентификатор бота (требуется для FSM StorageKey)."""
    bot_id = getattr(bot, "id", None)
    if not bot_id:
        bot_id = (await bot.me()).id
    return bot_id


async def _friend_profile_label(friend_tg: int) -> str:
    """Формирует человеко-понятное имя профиля друга для сообщений."""
    chat = await bot.get_chat(friend_tg)
    base = chat.full_name or chat.username or "друга"
    username = f" (@{chat.username})" if chat.username else ""
    return f"{base}{username}"


async def _start_friend_name_flow(
    user_tg_id: int,
    friend_ids: tuple[int, int],
    *,
    message: Message | None = None,
    state: FSMContext | None = None,
    intro: str | None = None,
) -> None:
    """Переводит пользователя в состояние задания имени другу и присылает подсказку."""

    sorted_ids = tuple(sorted(friend_ids))
    friend_tg = sorted_ids[1] if user_tg_id == sorted_ids[0] else sorted_ids[0]
    friend_label = await _friend_profile_label(friend_tg)

    if state is None or state.key.user_id != user_tg_id:
        bot_id = await _ensure_bot_id()
        state = FSMContext(storage=dp.storage, key=StorageKey(bot_id=bot_id, chat_id=user_tg_id, user_id=user_tg_id))

    await state.set_state(FriendStates.awaiting_friend_name)
    await state.update_data(friend_ids=sorted_ids, friend_tg_id=friend_tg, friend_label=friend_label)

    header = intro or f"👥 Новый друг добавлен!"
    text = (
        f"{header}\n"
        f"Напиши, как ты хочешь называть {friend_label} в боте.\n"
        "Без имени бот не сможет продолжить работу."
    )

    if message is not None:
        await message.answer(text)
    else:
        await bot.send_message(user_tg_id, text)


async def _find_pending_friend(user_tg_id: int, *, exclude: tuple[int, int] | None = None) -> asyncpg.Record | None:
    """Возвращает пару пользователей, для которой нужно задать имя другу."""

    async with db_pool.acquire() as conn:
        params: list[int] = [user_tg_id]
        exclude_cond = ""
        if exclude:
            x, y = sorted(exclude)
            exclude_cond = " AND NOT (user_a=$2 AND user_b=$3)"
            params.extend([x, y])

        query = f"""
            SELECT user_a, user_b
            FROM friends
            WHERE (user_a=$1 OR user_b=$1)
              AND status='accepted'
              AND ((user_a=$1 AND name_for_a IS NULL) OR (user_b=$1 AND name_for_b IS NULL))
              {exclude_cond}
            ORDER BY COALESCE(answered_at, requested_at, NOW()) DESC
            LIMIT 1
        """
        return await conn.fetchrow(query, *params)


async def ensure_friend_name_required(message: Message, state: FSMContext) -> bool:
    """Проверяет, нужно ли пользователю назвать друга, и при необходимости просит это сделать."""

    current_state = await state.get_state()
    tg_id = message.from_user.id

    if current_state == FriendStates.awaiting_friend_name.state:
        data = await state.get_data()
        friend_label = data.get("friend_label") or "нового друга"
        await message.answer(
            f"👥 Сначала назови {friend_label}. Просто отправь одно сообщение с именем.")
        return True

    pending = await _find_pending_friend(tg_id)
    if pending:
        await _start_friend_name_flow(
            tg_id,
            (pending["user_a"], pending["user_b"]),
            message=message,
            state=state,
            intro=None,
        )
        return True

    return False



@dp.message(F.text == "Друзья")
async def handle_friends_menu(message: Message, state: FSMContext):
    if await ensure_friend_name_required(message, state):
        return
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👥 Список друзей", callback_data="friends_list")],
        [InlineKeyboardButton(text="➕ Добавить друга", callback_data="friend_add")],
    ])
    await message.answer(
        "Здесь ты можешь управлять своими друзьями:\n"
        "— Просматривать список друзей\n"
        "— Приглашать новых\n\n"
        "После добавления — дай своему другу имя для удобства назначения задач 💬",
        reply_markup=kb
    )

# --- кнопка "Добавить друга" ---
@dp.callback_query(F.data == "friend_add")
async def friend_add_button(cb: CallbackQuery, state: FSMContext):
    if await ensure_friend_name_required(cb.message, state):
        await cb.answer()
        return
    inviter_id = cb.from_user.id

    # создаём простую (некодированную) deep-link и явную команду
    bot_username = (await bot.me()).username
    link = f"https://t.me/{bot_username}?start=friend_{inviter_id}"
    cmd = f"/start friend_{inviter_id}"

    # отправляем 2 сообщения:
    # 1) с объяснением и кнопкой
    await cb.message.answer(
        "🔗 <b>Как добавить друга?</b>\n"
        "1. Перешли это сообщение другу.\n"
        "2. Он <u>либо</u> нажмёт кнопку ниже,\n"
        "   <u>либо</u> скопирует команду из следующего сообщения и отправит её боту.\n\n"
        "После этого вы автоматически станете друзьями 💫",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="👥 Добавиться в друзья", url=link)]]
        ),
        parse_mode=ParseMode.HTML
    )

    await cb.answer()

# --- сохранение имени друга ---
@dp.message(FriendStates.awaiting_friend_name)
async def save_friend_name(message: Message, state: FSMContext):
    data = await state.get_data()
    tg_id = message.from_user.id
    friend_name = message.text.strip()

    if not friend_name:
        await message.answer("⚠️ Имя не может быть пустым. Напиши, как ты хочешь называть друга.")
        return

    async with db_pool.acquire() as conn:
        me = tg_id  # telegram_id = PK
        friend_ids = data.get("friend_ids")
        if not friend_ids:
            await message.answer("Ошибка. Попробуйте заново.")
            return

        user_a, user_b = sorted(friend_ids)

        await conn.execute(
            f"UPDATE friends SET {'name_for_a' if me == user_a else 'name_for_b'}=$1 WHERE user_a=$2 AND user_b=$3",
            friend_name, user_a, user_b
        )

        friend_id = user_b if me == user_a else user_a
        friend_tg = friend_id  # уже telegram_id

    friend_label = data.get("friend_label")
    if not friend_label:
        friend_label = await _friend_profile_label(friend_tg)

    await state.clear()
    await message.answer(
        f"✅ Имя друга сохранено!\nТы назвал(а) {friend_label}: <b>{friend_name}</b>",
        reply_markup=main_kb,
        parse_mode=ParseMode.HTML
    )

    next_pending = await _find_pending_friend(tg_id, exclude=tuple(sorted(friend_ids)))
    if next_pending:
        await _start_friend_name_flow(
            tg_id,
            (next_pending["user_a"], next_pending["user_b"]),
            message=message,
            state=state,
            intro="👥 Остались и другие друзья без имени.",
        )


########################################################################################################################
########### НОВЫЙ ФУНКЦИОНАЛ РАБОТЫ С ДРУЗЬЯМИ #######################################################################
#Хэндлер кнопки "Подтвердить" для друга
@dp.callback_query(F.data == "friend_task_confirm")
async def friend_task_choose_type(cb: CallbackQuery, state: FSMContext):
    await cb.message.edit_text("Какую задачу создать?")
    await state.set_state(FriendTaskStates.waiting_type)
    await cb.message.edit_reply_markup(
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="👫 Совместная задача", callback_data="friend_task_joint")],
            [InlineKeyboardButton(text="🎯 Только другу", callback_data="friend_task_target")],
        ])
    )
    await cb.answer()

#Обработка "Совместная задача" "Только другу"
@dp.callback_query(F.data.in_(["friend_task_joint", "friend_task_target"]))
async def handle_friend_task_type(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    title = data["title"]
    start_iso = data["start_iso"]
    end_iso = data["end_iso"] or None
    friend_id = data["friend_user_id"]        # это telegram_id
    task_type = "joint" if cb.data.endswith("joint") else "target"

    author_tg_id = cb.from_user.id            # это telegram_id
    async with db_pool.acquire() as conn:
        author_id = author_tg_id
        friend_tg_id = friend_id
        tz_offset = await conn.fetchval(
            "SELECT tz_offset_min FROM users_planner WHERE telegram_id=$1", author_tg_id
        ) or 3

        start_dt = datetime.fromisoformat(start_iso)
        end_dt = datetime.fromisoformat(end_iso) if end_iso else None

        # конфликт у автора (для совместной)
        if task_type == "joint":
            conflicts = await conn.fetch(
                CONFLICT_SQL, author_id, start_dt, end_dt or start_dt
            )
            if conflicts:
                old = conflicts[0]
                await state.update_data(
                    conflict_task_id=old["id"],
                    conflict_task_title=old["title"],
                    conflict_start=old["start_dt"].isoformat(),
                    conflict_end=(old["end_dt"].isoformat() if old["end_dt"] else ""),
                    all_day=False,
                    friend_user_id=friend_id,
                    type="joint"
                )
                await cb.message.edit_text("⚠️ У вас конфликт с другой задачей. Что делать?")
                await cb.message.edit_reply_markup(
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                        [InlineKeyboardButton(text="🕑 Передвинуть старую", callback_data="move_old")],
                        [InlineKeyboardButton(text="🕓 Передвинуть новую", callback_data="move_new")],
                        [InlineKeyboardButton(text="❌ Удалить старую", callback_data="delete_old")],
                    ])
                )
                await cb.answer()
                return

        user_a, user_b = sorted([author_id, friend_id])
        friend_name = await conn.fetchval(
            f"""
            SELECT { 'name_for_b' if author_id < friend_id else 'name_for_a' }
            FROM friends WHERE user_a=$1 AND user_b=$2
            """,
            user_a, user_b
        ) or "друг"

        time_str = start_dt.strftime("%d.%m %H:%M")
        if end_dt:
            time_str += f" — {end_dt.strftime('%H:%M')}"

        await conn.execute("""
            INSERT INTO pending_tasks (title, start_dt, end_dt, type, from_user_id, to_user_id)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, title, start_dt, end_dt, task_type, author_id, friend_id)

        await bot.send_message(
            chat_id=friend_tg_id,
            text=(f"📨 <b>{friend_name}</b> предлагает задачу:\n"
                  f"<b>{title}</b>\n🕒 {time_str}"),
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="✅ Принять", callback_data=f"accept_task_{task_type}")],
                [InlineKeyboardButton(text="❌ Отклонить", callback_data="reject_task")],
            ]),
            parse_mode=ParseMode.HTML
        )

    await state.clear()
    await cb.message.edit_text("📨 Задача отправлена другу!")
    await cb.answer()

# Хэндлер принятия задачи от друга
async def handle_accept_task(cb: CallbackQuery, state: FSMContext):
    task_type = cb.data.split("_", 2)[-1]  # joint | target
    user_tg_id = cb.from_user.id           # получатель (telegram_id)

    async with db_pool.acquire() as conn:
        to_user_id = user_tg_id

        task = await conn.fetchrow(
            "SELECT * FROM pending_tasks WHERE to_user_id = $1",
            to_user_id
        )
        if not task:
            await cb.message.edit_text("❌ Задача не найдена или уже удалена.")
            await cb.answer()
            return

        start_dt = task["start_dt"]
        end_dt   = task["end_dt"]
        title    = task["title"]
        author_id = task["from_user_id"]

        # конфликты у получателя
        conflicts = await conn.fetch(
            CONFLICT_SQL, to_user_id, start_dt, end_dt or start_dt
        )
        if conflicts:
            old = conflicts[0]
            await state.update_data(
                conflict_task_id=old["id"],
                conflict_task_title=old["title"],
                conflict_start=old["start_dt"].isoformat(),
                conflict_end=(old["end_dt"].isoformat() if old["end_dt"] else ""),
                friend_user_id=author_id,
                type=task_type,
                title=title,
                start_iso=start_dt.isoformat(),
                end_iso=end_dt.isoformat() if end_dt else "",
            )
            await cb.message.edit_text(
                f"⚠️ Задача пересекается с вашей задачей «{old['title']}». Что хотите сделать?"
            )
            await cb.message.edit_reply_markup(
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🕑 Передвинуть старую", callback_data="move_old")],
                    [InlineKeyboardButton(text="🕓 Передвинуть новую", callback_data="move_new")],
                    [InlineKeyboardButton(text="❌ Удалить старую", callback_data="delete_old")],
                ])
            )
            await cb.answer()
            return

        # создаём задачи
        if task_type == "target":
            # только у получателя
            await conn.execute(
                """
                INSERT INTO tasks (title, start_dt, end_dt,
                                   assigned_to_user_id, assigned_by_user_id, status)
                VALUES ($1, $2, $3, $4, $5, 'pending')
                """,
                title, start_dt, end_dt, to_user_id, author_id
            )
        else:
            # совместная: у обоих
            await conn.execute(
                """
                INSERT INTO tasks (title, start_dt, end_dt,
                                   assigned_to_user_id, assigned_by_user_id, status)
                VALUES ($1, $2, $3, $4, $5, 'pending')
                """,
                title, start_dt, end_dt, to_user_id, author_id
            )
            await conn.execute(
                """
                INSERT INTO tasks (title, start_dt, end_dt,
                                   assigned_to_user_id, assigned_by_user_id, status)
                VALUES ($1, $2, $3, $4, $5, 'pending')
                """,
                title, start_dt, end_dt, author_id, author_id
            )

        recipient_name = await conn.fetchval(
            "SELECT full_name FROM users_planner WHERE telegram_id = $1",
            to_user_id
        ) or "друг"

        author_tg_id = author_id
        time_str = start_dt.strftime("%d.%m %H:%M") + (f" — {end_dt.strftime('%H:%M')}" if end_dt else "")

        await bot.send_message(
            chat_id=author_tg_id,
            text=f"✅ <b>{recipient_name}</b> принял задачу:\n<b>{title}</b>\n🕒 {time_str}",
            parse_mode=ParseMode.HTML
        )

        await conn.execute("DELETE FROM pending_tasks WHERE id=$1", task["id"])

    await cb.message.edit_text("✅ Задача успешно добавлена в ваш список!")
    await cb.answer()


# Хэндлер отклонения задачи от друга
@dp.callback_query(F.data == "reject_task")
async def handle_reject_task(cb: CallbackQuery):
    user_tg_id = cb.from_user.id
    async with db_pool.acquire() as conn:
        task = await conn.fetchrow(
            "SELECT * FROM pending_tasks WHERE to_user_id=$1",
            user_tg_id
        )
        if task:
            await conn.execute("DELETE FROM pending_tasks WHERE id=$1", task["id"])

    await cb.message.edit_text("🚫 Вы отклонили задачу.")
    await cb.answer()















# --------------------------------------------------
async def main() -> None:
    await init_db()
    asyncio.create_task(reminders_worker())
    await dp.start_polling(bot)


@dp.message(StateFilter(None))
async def any_text(message: Message, state: FSMContext):
    if await ensure_friend_name_required(message, state):
        return
    uid = message.from_user.id

    async with db_pool.acquire() as conn:
        tz_offset = await conn.fetchval(
            "SELECT tz_offset_min FROM users_planner WHERE telegram_id=$1", uid
        ) or 3

    parsed = await gpt_parse(message.text, tz_offset)
    print(f"[GPT RESULT] Parsed: {parsed}")
    if not parsed:
        await message.answer(
            "❌ Не понял дату/время. Попробуй точнее 👉 «25 мая в 14:00 собрание»."
        )
        return

    # Оставляем timezone-aware (GPT уже вернул время с офсетом пользователя)
    start_dt = datetime.fromisoformat(parsed["start_iso"])
    end_dt = datetime.fromisoformat(parsed["end_iso"]) if parsed.get("end_iso") else None

    # Проверка на прошедшее время
    now = datetime.now().astimezone(start_dt.tzinfo)
    is_all_day = (not parsed.get("end_iso") and start_dt.time() == datetime.min.time())
    if not is_all_day and start_dt.date() == now.date() and start_dt.time() < now.time():
        await message.answer("❗️Время задачи уже прошло. Перепиши, пожалуйста, задачу на актуальное время.")
        return

    # --- Пытаемся найти имя друга в тексте ---
    async with db_pool.acquire() as conn:
        user_id = await conn.fetchval("SELECT telegram_id FROM users_planner WHERE telegram_id=$1", uid)
        friends = await conn.fetch("""
            SELECT 
                f.user_a, f.user_b,
                CASE WHEN f.user_a = $1 THEN f.name_for_a ELSE f.name_for_b END AS name,
                CASE WHEN f.user_a = $1 THEN f.user_b ELSE f.user_a END AS friend_id
            FROM friends f 
            WHERE (f.user_a = $1 OR f.user_b = $1) AND f.status = 'accepted'
        """, user_id)

    friend_match = None
    for row in friends:
        name = row["name"]
        if name and name.lower() in message.text.lower():
            friend_match = row
            break

    if friend_match:
        # Предлагаем задачу другу
        await state.update_data(
            title=parsed["title"],
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat() if end_dt else "",
            friend_user_id=friend_match["friend_id"],
            friend_name=friend_match["name"]
        )

        when = start_dt.strftime("%d.%m.%Y %H:%M")
        if end_dt:
            when += f" — {end_dt.strftime('%H:%M')}"

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="✅ Подтвердить", callback_data="friend_task_confirm")],
                [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")],
            ]
        )

        await message.answer(
            f"📌 Задача для <b>{friend_match['name']}</b>:\n"
            f"{parsed['title']}\n"
            f"{when}\n\nПодтвердить?",
            reply_markup=kb,
            parse_mode=ParseMode.HTML
        )
        return

    # --- Старый личный функционал, если имя друга не найдено ---
    await state.update_data(
        title=parsed["title"],
        start_iso=start_dt.isoformat(),
        end_iso=end_dt.isoformat() if end_dt else ""
    )

    when = start_dt.strftime("%d.%m.%Y %H:%M")
    if end_dt:
        when += f" — {end_dt.strftime('%H:%M')}"

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Подтвердить", callback_data="confirm_task")],
            [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")],
        ]
    )

    await message.answer(
        f"Создать задачу:\n<b>{parsed['title']}</b>\n{when}\nПодтверждаешь?",
        parse_mode=ParseMode.HTML,
        reply_markup=kb,
    )

### Функционал с Друзьями и челленджами

async def send_challenge_invites(ch_id: int, initiator_id: int):
    """
    Шлёт инвайты всем участникам (кроме создателя).
    Кнопки: Принять/Отклонить.
    """
    async with db_pool.acquire() as conn:
        ch = await conn.fetchrow("""
            SELECT id, title, type::text AS type, step_default, start_dt, end_dt
            FROM challenges WHERE id=$1
        """, ch_id)
        if not ch:
            return

        parts = await conn.fetch("""
            SELECT user_id, is_owner, is_accepted
            FROM challenge_participants
            WHERE challenge_id=$1
        """, ch_id)

    title = ch["title"]
    when = ""
    if ch["start_dt"]: when += ch["start_dt"].strftime("%d.%m.%Y")
    if ch["end_dt"]:   when += f" — {ch['end_dt'].strftime('%d.%m.%Y')}"

    for p in parts:
        uid = p["user_id"]
        if uid == initiator_id:  # автору не шлём инвайт
            continue
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="✅ Принять", callback_data=f"ch:acc:{ch_id}")],
            [InlineKeyboardButton(text="❌ Отклонить", callback_data=f"ch:rej:{ch_id}")],
        ])
        with suppress(Exception):
            await bot.send_message(
                chat_id=uid,
                text=f"🏆 Приглашение в челлендж:\n<b>{title}</b>\n{when}",
                reply_markup=kb,
                parse_mode=ParseMode.HTML,
            )

def challenge_action_kb(ch_id: int, ch_type: str, step: float | int | None = 1):
    rows = []
    if ch_type == "daily":
        rows.append([InlineKeyboardButton(text="✅ Сегодня сделано", callback_data=f"ch:ci:{ch_id}")])
    elif ch_type == "quant":
        # быстрые шаги + кастом
        step = step or 1
        rows.append([
            InlineKeyboardButton(text=f"+{int(step)}", callback_data=f"ch:qadd:{ch_id}:{int(step)}"),
            InlineKeyboardButton(text="+Другое", callback_data=f"ch:qadd:{ch_id}:ask"),
        ])
    elif ch_type == "event":
        rows.append([InlineKeyboardButton(text="➕ Визит", callback_data=f"ch:ev:{ch_id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

@dp.callback_query(F.data.regexp(r"^ch:acc:(\d+)$"))
async def ch_accept(cb: CallbackQuery):
    ch_id = int(cb.data.split(":")[-1])
    uid = cb.from_user.id
    async with db_pool.acquire() as conn:
        # пометить участника принятым
        updated = await conn.execute("""
            UPDATE challenge_participants
               SET is_accepted=true, joined_at=now()
             WHERE challenge_id=$1 AND user_id=$2
        """, ch_id, uid)
        # вытащим тип/шаг для кнопок
        ch = await conn.fetchrow("SELECT title, type::text AS type, step_default FROM challenges WHERE id=$1", ch_id)

    if not ch:
        await cb.message.edit_text("❌ Челлендж не найден.")
        return

    await cb.message.edit_text(
        f"✅ Вы присоединились к челленджу: <b>{ch['title']}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=challenge_action_kb(ch_id, ch["type"], ch["step_default"]),
    )
    await cb.answer()

@dp.callback_query(F.data.regexp(r"^ch:rej:(\d+)$"))
async def ch_reject(cb: CallbackQuery):
    ch_id = int(cb.data.split(":")[-1])
    uid = cb.from_user.id
    async with db_pool.acquire() as conn:
        await conn.execute("""
            DELETE FROM challenge_participants
             WHERE challenge_id=$1 AND user_id=$2 AND is_owner=false
        """, ch_id, uid)
    await cb.message.edit_text("🚫 Вы отклонили приглашение.")
    await cb.answer()

### Быстрые отметки прогресса в боте
@dp.callback_query(F.data.regexp(r"^ch:ci:(\d+)$"))
async def ch_checkin_today(cb: CallbackQuery):
    ch_id = int(cb.data.split(":")[-1])
    uid = cb.from_user.id
    async with db_pool.acquire() as conn:
        try:
            await conn.execute("""
                INSERT INTO challenge_progress (challenge_id, user_id, ts, value, source)
                VALUES ($1, $2, now(), 1, 'bot_push')
            """, ch_id, uid)
            title = await conn.fetchval("SELECT title FROM challenges WHERE id=$1", ch_id)
        except Exception as e:
            # триггер daily не даст отметить 2 раза за локальный день
            await cb.answer("Уже отмечено сегодня 👌", show_alert=True)
            return
    await cb.answer("Готово! ✅")
    with suppress(Exception):
        await cb.message.reply(f"✅ Отмечено в «{title}». Держим темп!")

@dp.callback_query(F.data.regexp(r"^ch:qadd:(\d+):(ask|\d+)$"))
async def ch_quant_add(cb: CallbackQuery, state: FSMContext):
    _, _, ch_id_s, how = cb.data.split(":")
    ch_id = int(ch_id_s)
    if how == "ask":
        await state.update_data(ch_qadd_id=ch_id)
        await cb.message.answer("Сколько добавить? Введите число:")
        await cb.answer()
        return
    value = int(how)
    await _do_qadd(cb, ch_id, value)

@dp.message(F.text.regexp(r"^\d+(\.\d+)?$"))
async def ch_quant_add_value(msg: Message, state: FSMContext):
    data = await state.get_data()
    ch_id = data.get("ch_qadd_id")
    if not ch_id:
        return
    with suppress(Exception):
        await _do_qadd(msg, ch_id, float(msg.text))
    await state.update_data(ch_qadd_id=None)

async def _do_qadd(obj: typing.Union[CallbackQuery, Message], ch_id: int, value: float):
    uid = (obj.from_user.id if isinstance(obj, CallbackQuery) else obj.from_user.id)
    async with db_pool.acquire() as conn:
        ch = await conn.fetchrow("SELECT title, type::text AS type FROM challenges WHERE id=$1", ch_id)
        if not ch or ch["type"] != "quant":
            if isinstance(obj, CallbackQuery):
                await obj.answer("Неверный тип челленджа", show_alert=True); return
            else:
                await obj.answer("Неверный тип челленджа"); return
        await conn.execute("""
            INSERT INTO challenge_progress (challenge_id, user_id, ts, value, source)
            VALUES ($1,$2,now(),$3,'bot_push')
        """, ch_id, uid, value)
    txt = f"➕ Добавлено {value} в «{ch['title']}»"
    if isinstance(obj, CallbackQuery):
        await obj.answer("Готово!")
        await obj.message.reply(txt)
    else:
        await obj.answer(txt)

@dp.callback_query(F.data.regexp(r"^ch:ev:(\d+)$"))
async def ch_event_add(cb: CallbackQuery):
    ch_id = int(cb.data.split(":")[-1])
    uid = cb.from_user.id
    async with db_pool.acquire() as conn:
        ch = await conn.fetchrow("SELECT title, type::text AS type FROM challenges WHERE id=$1", ch_id)
        if not ch or ch["type"] != "event":
            await cb.answer("Неверный тип челленджа", show_alert=True); return
        await conn.execute("""
            INSERT INTO challenge_progress (challenge_id, user_id, ts, value, source)
            VALUES ($1,$2,now(),1,'bot_push')
        """, ch_id, uid)
    await cb.answer("Засчитано! ➕")

### Команда для списка челленджей (быстрый доступ из бота)
@dp.message(Command("challenges"))
async def list_my_challenges(message: Message, state: FSMContext):
    if await ensure_friend_name_required(message, state):
        return
    uid = message.from_user.id
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT c.id, c.title, c.type::text AS type, c.step_default
            FROM challenges c
            JOIN challenge_participants p ON p.challenge_id=c.id
            WHERE p.user_id=$1 AND p.is_accepted=true AND c.archived_at IS NULL
            ORDER BY c.start_dt NULLS LAST, c.id DESC
        """, uid)
    if not rows:
        await message.answer("Пока нет активных челленджей. Создайте в мини-приложении 🏆")
        return
    for r in rows:
        kb = challenge_action_kb(r["id"], r["type"], r["step_default"])
        await message.answer(f"🏆 <b>{r['title']}</b>", parse_mode=ParseMode.HTML, reply_markup=kb)

#### Фоновая задача-напоминалка (локальное время, без спама)

async def reminders_worker():
    """
    Каждую минуту:
      - выбираем челленджи с remind_at_local
      - для каждого участника считаем его локальное hh:mm
      - если совпадает и не отправляли сегодня — шлём пуш с кнопкой действия
    """
    await ensure_aux_tables()
    while True:
        try:
            async with db_pool.acquire() as conn:
                chs = await conn.fetch("""
                    SELECT id, title, type::text AS type, step_default, remind_at_local, remind_days
                    FROM challenges
                    WHERE remind_at_local IS NOT NULL AND archived_at IS NULL
                """)
                for ch in chs:
                    ch_id = ch["id"]
                    remind_time: datetime.time = ch["remind_at_local"]
                    days = ch["remind_days"]  # может быть NULL
                    parts = await conn.fetch("""
                        SELECT p.user_id, u.tz_offset_min
                        FROM challenge_participants p
                        JOIN users_planner u ON u.telegram_id=p.user_id
                        WHERE p.challenge_id=$1 AND p.is_accepted=true
                    """, ch_id)
                    for p in parts:
                        uid = p["user_id"]
                        off = p["tz_offset_min"] or 0
                        now_loc = now_local(off)
                        # фильтр по дням (если задан)
                        if days and now_loc.weekday() not in [d if d != 0 else 6 for d in days]:
                            # Примечание: если ты хранишь вс=0, пн=1... отрегулируй
                            pass
                        hhmm_ok = (now_loc.hour == remind_time.hour and now_loc.minute == remind_time.minute)
                        if not hhmm_ok:
                            continue
                        # уже отправляли сегодня?
                        sent = await conn.fetchval("""
                            SELECT 1 FROM challenge_reminders_sent
                            WHERE challenge_id=$1 AND user_id=$2 AND local_day=$3
                        """, ch_id, uid, now_loc.date())
                        if sent:
                            continue
                        # отправка
                        kb = challenge_action_kb(ch_id, ch["type"], ch["step_default"])
                        with suppress(Exception):
                            await bot.send_message(uid, f"🔔 Напоминание по челленджу «{ch['title']}»", reply_markup=kb)
                        await conn.execute("""
                            INSERT INTO challenge_reminders_sent(challenge_id, user_id, local_day)
                            VALUES ($1,$2,$3) ON CONFLICT DO NOTHING
                        """, ch_id, uid, now_loc.date())
        except Exception as e:
            logging.exception("reminders_worker error")
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
