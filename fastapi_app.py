from fastapi import FastAPI
from socials.webhook import router as socials_router

app = FastAPI(title="MetaForge FastAPI")

# Include routers
app.include_router(socials_router, prefix="/api/socials/webhook")

@app.get("/")
async def root():
    return {"message": "FastAPI is running"}
