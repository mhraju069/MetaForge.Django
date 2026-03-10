from fastapi import FastAPI
from socials.views import router as socials_router
from socials.webhook import router as webhook_router

app = FastAPI(title="MetaForge FastAPI")

# Include routers
# Note: socials_router handles items like /facebook/connect/, etc.
# Note: webhook_router handles /webhook/{platform}/
app.include_router(socials_router, prefix="/api/socials")
app.include_router(webhook_router, prefix="/api/socials")

@app.get("/")
async def root():
    return {"message": "FastAPI is running"}
