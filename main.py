import asyncio
import uvicorn
from bot.webhook_api import app as fastapi_app  # FastAPI-приложение
import config
from bot import main
from bot.bot import init_db


async def start_api():
    """Запускаем FastAPI на 8000"""
    config_uvicorn = uvicorn.Config(
        app=fastapi_app,
        host="0.0.0.0",
        port=8000,  # ← стандартный порт для API
        loop="asyncio",
        log_level="info"
    )
    server = uvicorn.Server(config_uvicorn)
    await server.serve()


async def main_task():
    """Основная корутина: бот + API"""
    await init_db()  # создаём пул БД один раз при старте

    if bool(config.ENABLE_RENDER):
        await asyncio.gather(
            main(),       # запуск бота
            start_api(),  # запуск API
        )
    else:
        await asyncio.gather(
            main(),
            start_api(),
        )


if __name__ == "__main__":
    import logging, sys, traceback
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except Exception:
        logging.exception("BOT CRASHED ON STARTUP")
        sys.exit(1)