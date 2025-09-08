from fastapi import FastAPI
from api.health.router import router as health_router
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.include_router(health_router)



if __name__ == "__main__":
    port = int(os.getenv("PORT", 74))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", workers=2)
