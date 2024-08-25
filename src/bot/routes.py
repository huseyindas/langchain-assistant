import logging

from fastapi import APIRouter, BackgroundTasks

from src.assistant.loader import Loader
from src.assistant.chain import Chain
from src.bot.schemas import ChatRequest
from src.core.consts import DOCUMENTS_PATH


router = APIRouter(prefix="/bot")


@router.post("/chat")
async def chat(chat_request: ChatRequest):
    chain = Chain(prompt_path="./prompt.txt")
    response = chain.chat(chat_request.message)
    logging.warn(str(chain.prompt))
    return response


@router.post("/load")
def load(background_tasks: BackgroundTasks):
    loader = Loader(
        documents_path=DOCUMENTS_PATH
    )
    background_tasks.add_task(loader.commit(), [""])
    return True
