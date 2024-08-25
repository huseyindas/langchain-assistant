from fastapi import FastAPI

from src.bot.routes import router


app = FastAPI()
app.include_router(router)


@app.get("/")
async def health():
    return {"status": True}
