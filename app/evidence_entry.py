"""Start as: python -m uvicorn app.evidence_entry:app --reload --port 8000"""
from fastapi import APIRouter
from app.mission_api import app
from nimbo.unified_evidence import collect

router=APIRouter()
@router.get('/nimbo/evidence/unified')
def unified_evidence(limit:int=100):
    return collect(limit=limit)
app.include_router(router)
