from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, analyze

app = FastAPI(title="Artificial Troubleshooter API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analyze.router)
