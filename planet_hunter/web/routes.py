from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from planet_hunter import db
from planet_hunter.models import QueueSource, Classification
from planet_hunter.config import PRIORITY_MANUAL

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    stats = db.count_by_classification()
    queue = db.queue_stats()
    recent = db.list_analyses(limit=10)
    ml_snapshot = db.ml_monitor_snapshot(hours=24)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats,
        "queue": queue,
        "recent": recent,
        "ml_snapshot": ml_snapshot,
    })


@router.get("/submit", response_class=HTMLResponse)
async def submit_form(request: Request):
    return templates.TemplateResponse("submit.html", {"request": request})


@router.post("/submit")
async def submit_tic(request: Request, tic_id: int = Form(...)):
    db.enqueue(tic_id, QueueSource.MANUAL, PRIORITY_MANUAL)
    return RedirectResponse(url="/queue", status_code=303)


@router.get("/results", response_class=HTMLResponse)
async def results_list(
    request: Request,
    classification: str = None,
    min_ml_score: float | None = None,
    model_version: str | None = None,
):
    analyses = db.list_analyses(
        classification=classification,
        limit=100,
        min_ml_score=min_ml_score,
        model_version=model_version,
    )
    classifications = list(Classification)
    return templates.TemplateResponse("results.html", {
        "request": request,
        "analyses": analyses,
        "current_filter": classification,
        "current_min_ml_score": min_ml_score,
        "current_model_version": model_version,
        "classifications": classifications,
    })


@router.get("/results/{analysis_id}", response_class=HTMLResponse)
async def result_detail(request: Request, analysis_id: int):
    analysis = db.get_analysis(analysis_id)
    if not analysis:
        return HTMLResponse("Not found", status_code=404)
    star = db.get_star(analysis["tic_id"])
    return templates.TemplateResponse("result_detail.html", {
        "request": request,
        "a": analysis,
        "star": star,
    })


@router.get("/queue", response_class=HTMLResponse)
async def queue_view(request: Request):
    items = db.list_queue(limit=100)
    stats = db.queue_stats()
    return templates.TemplateResponse("queue.html", {
        "request": request,
        "items": items,
        "stats": stats,
    })


@router.get("/review", response_class=HTMLResponse)
async def review_list(request: Request):
    analyses = db.list_analyses(classification="MANUAL_REVIEW", limit=100)
    return templates.TemplateResponse("review.html", {
        "request": request,
        "analyses": analyses,
    })


@router.post("/review/{analysis_id}")
async def review_submit(
    request: Request,
    analysis_id: int,
    classification: str = Form(...),
    notes: str = Form(""),
):
    db.update_analysis(
        analysis_id,
        classification=classification,
        review_notes=notes,
    )
    return RedirectResponse(url="/review", status_code=303)


# --- JSON API for live refresh ---

@router.get("/api/dashboard")
async def api_dashboard():
    return {
        "stats": db.count_by_classification(),
        "queue": db.queue_stats(),
        "recent": db.list_analyses(limit=10),
        "ml_snapshot": db.ml_monitor_snapshot(hours=24),
    }


@router.get("/api/queue")
async def api_queue():
    return {
        "items": db.list_queue(limit=100),
        "stats": db.queue_stats(),
    }


@router.get("/api/results")
async def api_results(
    classification: str = None,
    min_ml_score: float | None = None,
    model_version: str | None = None,
):
    return {
        "analyses": db.list_analyses(
            classification=classification,
            limit=100,
            min_ml_score=min_ml_score,
            model_version=model_version,
        ),
    }
