# bot/webhook_api.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
import asyncio 
import asyncpg

from db_config import DB_CONFIG
from bot.bot import db_pool, bot

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# если API запускается отдельно от бота — поднимем локальный пул
@app.on_event("startup")
async def _maybe_init_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(**DB_CONFIG)
        print("✅ DB Pool initialized (webhook_api)!")

# ---------- модели ----------
class TaskCreateRequest(BaseModel):
    telegram_id: int                 # кто создаёт (assigned_by_user_id)
    title: str
    description: str | None = None
    start_dt: datetime | None = None
    end_dt: datetime | None = None
    all_day: bool = False
    for_user: int | None = None      # кому назначаем (assigned_to_user_id)

# ---------- утилиты ----------
def local_day_bounds_utc(day_iso: str, offset_min: int) -> tuple[datetime, datetime]:
    """
    day_iso: 'YYYY-MM-DD' — дата, выбранная в календаре мини-приложения.
    offset_min: users_planner.tz_offset_min (в минутах).
    Возвращает (start_utc, end_utc) — границы суток в UTC.
    """
    y, m, d = map(int, day_iso.split("-"))
    local_tz = timezone(timedelta(minutes=offset_min))
    start_local = datetime(y, m, d, 0, 0, 0, tzinfo=local_tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

# ---------- endpoints ----------
@app.get("/")
async def root():
    return {"ok": True}

@app.get("/timezone")
async def get_timezone(uid: int = Query(...)):
    """Вернуть смещение пользователя в минутах."""
    async with db_pool.acquire() as conn:
        off = await conn.fetchval(
            "SELECT tz_offset_min FROM users_planner WHERE telegram_id=$1",
            uid,
        )
    return {"offset_min": off or 180}

@app.get("/tasks")
async def get_tasks(uid: int = Query(...), date: str = Query(...)):
    """
    Вернуть задачи пользователя (uid — это telegram_id),
    попадающие в выбранные локальные сутки.
    """
    try:
        async with db_pool.acquire() as conn:
            offset_min = await conn.fetchval(
                "SELECT tz_offset_min FROM users_planner WHERE telegram_id=$1",
                uid,
            ) or 0

            start_utc, end_utc = local_day_bounds_utc(date, offset_min)

            rows = await conn.fetch(
                """
                SELECT
                  t.id, t.title, t.description, t.start_dt, t.end_dt, t.all_day,
                  t.status::text AS status,
                  t.assigned_by_user_id AS author_id,
                  t.assigned_to_user_id AS assignee_id,
                  CASE
                    WHEN t.assigned_by_user_id IS NULL OR t.assigned_by_user_id = $1 THEN NULL
                    ELSE COALESCE(
                      (SELECT f.name_for_a
                         FROM friends f
                        WHERE f.user_a = $1 AND f.user_b = t.assigned_by_user_id AND f.status = 'accepted'
                        LIMIT 1),
                      (SELECT f.name_for_b
                         FROM friends f
                        WHERE f.user_b = $1 AND f.user_a = t.assigned_by_user_id AND f.status = 'accepted'
                        LIMIT 1),
                      (SELECT up.full_name
                         FROM users_planner up
                        WHERE up.telegram_id = t.assigned_by_user_id
                        LIMIT 1)
                    )
                  END AS from_name
                FROM tasks t
                WHERE
                  t.assigned_to_user_id = $1
                  AND t.deleted_at IS NULL
                  AND t.start_dt >= $2 AND t.start_dt < $3
                ORDER BY t.start_dt NULLS LAST, t.id
                """,
                uid, start_utc, end_utc,
            )

        return [{
            "id": r["id"],
            "title": r["title"],
            "description": r["description"],
            "start_dt": r["start_dt"].isoformat() if r["start_dt"] else None,
            "end_dt": r["end_dt"].isoformat() if r["end_dt"] else None,
            "all_day": bool(r["all_day"]),
            "status": r["status"] or "pending",
            "from_name": r["from_name"],  # «от кого»
        } for r in rows]

    except Exception as e:
        return {"error": str(e)}

@app.post("/add_task")
async def add_task(task: TaskCreateRequest):
    """
    Создать задачу.
    telegram_id – кто назначает (assigned_by_user_id),
    for_user (или сам telegram_id) – кому (assigned_to_user_id).
    """
    author = task.telegram_id
    assignee = task.for_user or author

    # duration_min — по возможности посчитаем на стороне API
    duration_min = None
    if task.start_dt and task.end_dt:
        duration_min = int((task.end_dt - task.start_dt).total_seconds() // 60)

    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO tasks (
                title, description, start_dt, end_dt, all_day,
                assigned_to_user_id, assigned_by_user_id,
                status, duration_min
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,'pending',$8)
            """,
            task.title, task.description, task.start_dt, task.end_dt, task.all_day,
            assignee, author, duration_min
        )
    return {"status": "success"}

### Функционал с друзьями и челленджами: #####

### Модель для создания челленджа и прогресса
class ChallengeCreate(BaseModel):
    title: str
    type: str                   # 'daily' | 'quant' | 'event'
    start_dt: datetime | None = None
    end_dt: datetime | None = None
    target_value: float | None = None
    unit: str | None = None
    step: float | None = 1
    allow_grace: bool | None = False
    auto_tag: str | None = None
    proof: str | None = "none"
    reminder: dict | None = None   # {"time_local":"19:30","days":"daily|weekdays"}
    participants: list[int]        # telegram_id, включая автора
    created_by: int                # telegram_id

class ChallengeProgressIn(BaseModel):
    value: float = 1
    ts: datetime | None = None
    source: str = "webapp"

###Дружественные списки
@app.get("/friends")
async def friends(uid: int = Query(...)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
              CASE WHEN f.user_a=$1 THEN f.user_b ELSE f.user_a END AS friend_id,
              COALESCE(
                CASE WHEN f.user_a=$1 THEN f.name_for_b ELSE f.name_for_a END,
                u.full_name
              ) AS full_name,
              u.tz_offset_min
            FROM friends f
            JOIN users_planner u
              ON u.telegram_id = CASE WHEN f.user_a=$1 THEN f.user_b ELSE f.user_a END
            WHERE (f.user_a=$1 OR f.user_b=$1) AND f.status='accepted'
            ORDER BY full_name NULLS LAST
        """, uid)
    return [{"user_id": r["friend_id"], "full_name": r["full_name"], "tz_offset_min": r["tz_offset_min"]} for r in rows]


@app.get("/friends/with-shared")
async def friends_with_shared(uid: int = Query(...), window_days: int = 14):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            WITH fr AS (
              SELECT CASE WHEN f.user_a=$1 THEN f.user_b ELSE f.user_a END AS fid
              FROM friends f
              WHERE (f.user_a=$1 OR f.user_b=$1) AND f.status='accepted'
            ), t AS (
              SELECT DISTINCT
                CASE WHEN t.assigned_by_user_id=$1 THEN t.assigned_to_user_id ELSE t.assigned_by_user_id END AS fid
              FROM tasks t
              WHERE
                (t.assigned_to_user_id=$1 OR t.assigned_by_user_id=$1)
                AND (t.assigned_to_user_id IN (SELECT fid FROM fr) OR t.assigned_by_user_id IN (SELECT fid FROM fr))
                AND t.deleted_at IS NULL
                AND (t.start_dt >= now() - interval '1 day'
                     AND t.start_dt <  now() + make_interval(days => $2))   -- ✅ вместо ($2 || ' days')::interval
            )
            SELECT u.telegram_id AS friend_id, u.full_name
            FROM t JOIN users_planner u ON u.telegram_id=t.fid
        """, uid, window_days)
    return [{"user_id": r["friend_id"], "full_name": r["full_name"]} for r in rows]

### Совместные задачи (короткий фид)
@app.get("/tasks/shared")
async def shared_tasks(uid: int = Query(...), window_days: int = 7):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            WITH fr AS (
              SELECT CASE WHEN f.user_a=$1 THEN f.user_b ELSE f.user_a END AS fid
              FROM friends f
              WHERE (f.user_a=$1 OR f.user_b=$1) AND f.status='accepted'
            )
            SELECT
              t.id, t.title, t.start_dt, t.end_dt,
              CASE WHEN t.assigned_by_user_id=$1 THEN t.assigned_to_user_id ELSE t.assigned_by_user_id END AS friend_id
            FROM tasks t
            WHERE t.deleted_at IS NULL
              AND ( (t.assigned_to_user_id=$1 AND t.assigned_by_user_id IN (SELECT fid FROM fr))
                 OR (t.assigned_by_user_id=$1 AND t.assigned_to_user_id IN (SELECT fid FROM fr)) )
              AND (t.start_dt >= now() - interval '1 day'
                   AND t.start_dt <  now() + make_interval(days => $2))   -- ← ВАЖНО
            ORDER BY t.start_dt
        """, uid, window_days)
        # подтянем имена друзей
        friend_ids = list({r["friend_id"] for r in rows})
        names = {}
        if friend_ids:
            rs = await conn.fetch("SELECT telegram_id, full_name FROM users_planner WHERE telegram_id = ANY($1::bigint[])", friend_ids)
            names = {r["telegram_id"]: r["full_name"] for r in rs}
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "title": r["title"],
            "start_dt": r["start_dt"].isoformat() if r["start_dt"] else None,
            "end_dt": r["end_dt"].isoformat() if r["end_dt"] else None,
            "participants": [
                {"user_id": uid},
                {"user_id": r["friend_id"], "full_name": names.get(r["friend_id"])}
            ]
        })
    return out


### Получить активные челленджи пользователя
@app.get("/challenges")
async def get_challenges(uid: int = Query(...)):
    async with db_pool.acquire() as conn:
        chs = await conn.fetch("""
            SELECT c.id, c.title, c.type::text AS type, c.start_dt, c.end_dt,
                   c.target_value, c.unit, c.step_default
            FROM challenges c
            JOIN challenge_participants p ON p.challenge_id=c.id
            WHERE p.user_id=$1 AND p.is_accepted=true AND c.archived_at IS NULL
            ORDER BY c.start_dt NULLS LAST, c.id DESC
        """, uid)
        # участники
        parts = await conn.fetch("""
            SELECT p.challenge_id, p.user_id, u.full_name
            FROM challenge_participants p
            JOIN users_planner u ON u.telegram_id=p.user_id
            WHERE p.challenge_id = ANY($1::bigint[]) AND p.is_accepted=true
        """, [ [r["id"] for r in chs] ] if chs else [[]])
        # прогресс (агрегаты)
        prog = await conn.fetch("""
            SELECT challenge_id, user_id, SUM(value) AS s
            FROM challenge_progress
            WHERE challenge_id = ANY($1::bigint[])
            GROUP BY challenge_id, user_id
        """, [ [r["id"] for r in chs] ] if chs else [[]])

    parts_by_ch = {}
    for r in parts:
        parts_by_ch.setdefault(r["challenge_id"], []).append({"user_id": r["user_id"], "full_name": r["full_name"]})
    prog_by_ch = {}
    for r in prog:
        prog_by_ch.setdefault(r["challenge_id"], {})[r["user_id"]] = float(r["s"] or 0)

    out = []
    for c in chs:
        out.append({
            "id": c["id"],
            "title": c["title"],
            "type": c["type"],
            "start_dt": c["start_dt"].isoformat() if c["start_dt"] else None,
            "end_dt": c["end_dt"].isoformat() if c["end_dt"] else None,
            "target_value": float(c["target_value"] or 0),
            "unit": c["unit"],
            "step_default": float(c["step_default"] or 1),
            "participants": parts_by_ch.get(c["id"], []),
            "progress": {"byUser": prog_by_ch.get(c["id"], {}), "target": float(c["target_value"] or 0)}
        })
    return out

### Создание челленджа (+ рассылка инвайтов)
@app.post("/challenges")
async def create_challenge(data: ChallengeCreate):
    if data.type not in ("daily","quant","event"):
        return {"error":"invalid type"}
    participants = list(dict.fromkeys(data.participants or []))  # uniq, preserve order
    if data.created_by not in participants:
        participants.insert(0, data.created_by)

    # reminders
    remind_at = None
    remind_days = None
    if data.reminder and data.reminder.get("time_local"):
        hh, mm = map(int, data.reminder["time_local"].split(":"))
        remind_at = datetime(2000,1,1,hh,mm).time()
        if data.reminder.get("days") == "weekdays":
            remind_days = [1,2,3,4,5]  # пн..пт (подстроишь при своей нумерации)
    scope = "friends" if len(participants) <= 5 else "group"

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            ch_id = await conn.fetchval("""
                INSERT INTO challenges
                    (title, type, scope, created_by, start_dt, end_dt, target_value, unit,
                     step_default, allow_grace, auto_tag, proof, remind_at_local, remind_days)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                RETURNING id
            """,
            data.title, data.type, scope, data.created_by,
            data.start_dt, data.end_dt, data.target_value, data.unit,
            data.step, data.allow_grace, data.auto_tag, data.proof,
            remind_at, remind_days)

            # участники
            for uid in participants:
                await conn.execute("""
                    INSERT INTO challenge_participants (challenge_id, user_id, is_owner, is_accepted)
                    VALUES ($1,$2,$3,$4)
                """, ch_id, uid, uid==data.created_by, uid==data.created_by)

    # шлём инвайты из API (есть bot)
    try:
        from bot.bot import send_challenge_invites as _send  # если ты оставишь функцию в bot.py как есть
    except Exception:
        _send = None
    if _send:
        asyncio.create_task(_send(ch_id, data.created_by))  # не блокируем HTTP

    return {"id": ch_id}

### Добавить прогресс по челленджу
@app.post("/challenges/{ch_id}/progress")
async def add_progress(ch_id: int, body: ChallengeProgressIn, uid: int = Query(...)):
    async with db_pool.acquire() as conn:
        try:
            await conn.execute("""
                INSERT INTO challenge_progress (challenge_id, user_id, ts, value, source)
                VALUES ($1,$2,$3,$4,$5)
            """, ch_id, uid, body.ts or datetime.utcnow().replace(tzinfo=timezone.utc), body.value, body.source)
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": True}

### Сводка по челленджу
@app.get("/challenges/{ch_id}/summary")
async def challenge_summary(ch_id: int):
    async with db_pool.acquire() as conn:
        ch = await conn.fetchrow("""
            SELECT id, title, type::text AS type, target_value, start_dt, end_dt
            FROM challenges WHERE id=$1
        """, ch_id)
        if not ch:
            return {"error":"not found"}
        parts = await conn.fetch("""
            SELECT p.user_id, u.full_name FROM challenge_participants p
            JOIN users_planner u ON u.telegram_id=p.user_id
            WHERE p.challenge_id=$1 AND p.is_accepted=true
        """, ch_id)
        prog = await conn.fetch("""
            SELECT user_id, local_day, SUM(value) AS v
            FROM challenge_progress
            WHERE challenge_id=$1
            GROUP BY user_id, local_day
            ORDER BY local_day
        """, ch_id)
    return {
        "challenge": {
            "id": ch["id"], "title": ch["title"], "type": ch["type"],
            "target_value": float(ch["target_value"] or 0),
            "start_dt": ch["start_dt"].isoformat() if ch["start_dt"] else None,
            "end_dt": ch["end_dt"].isoformat() if ch["end_dt"] else None,
        },
        "participants": [{"user_id": r["user_id"], "full_name": r["full_name"]} for r in parts],
        "progress_by_day": [
            {"user_id": r["user_id"], "local_day": r["local_day"].isoformat(), "value": float(r["v"])}
            for r in prog
        ]
    }

### Лента друзей
@app.get("/activity/friends")
async def friends_activity(uid: int = Query(...), limit: int = 50):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            WITH fr AS (
              SELECT CASE WHEN f.user_a=$1 THEN f.user_b ELSE f.user_a END AS friend_id
              FROM friends f
              WHERE (f.user_a=$1 OR f.user_b=$1) AND f.status='accepted'
            )
            SELECT cp.ts, cp.user_id, u.full_name, c.title AS challenge_title, c.type::text AS type, cp.value
            FROM challenge_progress cp
            JOIN challenges c ON c.id=cp.challenge_id
            JOIN users_planner u ON u.telegram_id=cp.user_id
            WHERE cp.user_id IN (SELECT friend_id FROM fr)
            ORDER BY cp.ts DESC
            LIMIT $2
        """, uid, limit)
    return [{
        "ts": r["ts"].isoformat(),
        "user_id": r["user_id"],
        "full_name": r["full_name"],
        "challenge_title": r["challenge_title"],
        "type": r["type"],
        "value": float(r["value"])
    } for r in rows]

