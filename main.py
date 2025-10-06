import asyncio
import uvicorn
from bot.webhook_api import app as fastapi_app  # FastAPI-приложение
import config
from bot import main



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

def _is_render_enabled() -> bool:
    """Корректно преобразуем переменную окружения к bool."""

    value = getattr(config, "ENABLE_RENDER", False)
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in {"1", "true", "yes", "on"}


async def main_task():
    """Основная корутина: бот + API"""
    if _is_render_enabled():
        await asyncio.gather(
            main(),       # запуск бота
            start_api(),  # запуск API
        )
    else:
        await main()


if __name__ == "__main__":
    import logging, sys, traceback
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main_task())
    except Exception:
        logging.exception("BOT CRASHED ON STARTUP")
        sys.exit(1)