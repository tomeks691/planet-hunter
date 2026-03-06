import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from planet_hunter import db
from planet_hunter.config import PLOT_DIR
from planet_hunter.pipeline.runner import PipelineRunner
from planet_hunter.scanner.auto_scanner import AutoScanner
from planet_hunter.web.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

pipeline_runner = PipelineRunner()
auto_scanner = AutoScanner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    pipeline_runner.start()
    auto_scanner.start()
    logging.getLogger(__name__).info("Planet Hunter started")
    yield
    pipeline_runner.stop()
    auto_scanner.stop()
    logging.getLogger(__name__).info("Planet Hunter shutting down")


app = FastAPI(title="Planet Hunter", lifespan=lifespan)

# Static files
static_dir = Path(__file__).parent / "web" / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Plot images
app.mount("/plots", StaticFiles(directory=str(PLOT_DIR)), name="plots")

# Web routes
app.include_router(router)


# API: scanner toggle
@app.post("/api/scanner/toggle")
async def toggle_scanner():
    running = auto_scanner.toggle()
    return {"running": running}


@app.get("/api/scanner/status")
async def scanner_status():
    return {
        "scanner_running": auto_scanner.running,
        "pipeline_running": pipeline_runner.running,
    }
