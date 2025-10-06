from aiogram.filters import Filter
from aiogram.types import Message
import config

class AllowedIdFilter(Filter):
    """
    Filter for Telegram bot to check whether user
    is able to interact with bot
    """

    def __init__(self) -> None:
        # Проверка на пустоту или None
        if config.ALLOWED_IDS is None or config.ALLOWED_IDS == '*':
            self.allowed_ids = None  # Если пусто, разрешаем всем
        else:
            self.allowed_ids = list(map(int, config.ALLOWED_IDS.split(',')))

    async def __call__(self, message: Message) -> bool:
        """
        Filter checks whether the incoming message from user
        allowed to interact with the bot
        :param message: incoming message
        :return: true if user allowed to interact with bot, false otherwise
        """
        if self.allowed_ids is None:
            return True  # Разрешаем всем пользователям
        return message.from_user.id in self.allowed_ids

