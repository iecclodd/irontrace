from contextlib import asynccontextmanager
from pathlib import Path
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from .runtime import Runtime, StateError
from .schemas import NewSession, Mutation, Acquire

def create_app(runtime=None):
    @asynccontextmanager
    async def lifespan(app):
        if app.state.runtime is None:
            app.state.runtime=Runtime()
            app.state.runtime.jobs.submit(lambda progress: app.state.runtime.initialize())
        yield
        app.state.runtime.jobs.close()
    app=FastAPI(title='IRONTRACE',description='by azaan noman — local hydraulic inspection replay',version='0.1.0',lifespan=lifespan)
    app.state.runtime=runtime
    app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173','http://127.0.0.1:8000'],
                       allow_methods=['GET','POST'],allow_headers=['Content-Type'])

    @app.exception_handler(StateError)
    async def state_error(request,exc):
        return JSONResponse(status_code=409,content={'detail':str(exc)})
    @app.exception_handler(KeyError)
    async def missing(request,exc):
        return JSONResponse(status_code=404,content={'detail':'Resource not found.'})
    @app.exception_handler(RuntimeError)
    async def unavailable(request,exc):
        return JSONResponse(status_code=503,content={'detail':str(exc)})
    def rt(): return app.state.runtime

    @app.get('/api/health')
    def health(): return rt().health()
    @app.get('/api/model-info')
    def model_info(): return rt().provenance()
    @app.get('/api/demo-manifest')
    def manifest(): return rt().manifest
    @app.post('/api/sessions')
    def create(body:NewSession): return rt().create(body)
    @app.get('/api/sessions/{sid}')
    def session(sid:str): return rt().get(sid)
    @app.post('/api/sessions/{sid}/recommend')
    def recommend(sid:str,body:Mutation): return {'job_id':rt().queue_recommend(sid,body.version)}
    @app.post('/api/sessions/{sid}/acquire')
    def acquire(sid:str,body:Acquire): return rt().acquire(sid,body)
    @app.post('/api/sessions/{sid}/run')
    def run(sid:str,body:Mutation): return {'job_id':rt().queue_run(sid,body.version)}
    @app.post('/api/sessions/{sid}/stop')
    def stop(sid:str,body:Mutation): return rt().stop(sid,body.version)
    @app.get('/api/jobs/{jid}')
    def job(jid:str): return rt().jobs.get(jid)
    @app.get('/api/sessions/{sid}/trace')
    def trace(sid:str):
        rt().get(sid)
        return {'events':rt().store.events(sid)}
    @app.get('/api/sessions/{sid}/report')
    def report(sid:str): return rt().report(sid)
    @app.get('/api/benchmarks')
    def benchmarks(): return rt().benchmarks()
    @app.get('/api/replays')
    def replays(): return rt().replays()
    @app.get('/api/replays/{rid}')
    def replay(rid:str):
        allowed={x['id'] for x in rt().replays()['replays']}
        if rid not in allowed: raise HTTPException(404,'Recorded replay not found.')
        from .replay.public import public_replay
        return public_replay(json.loads((rt().settings.artifacts/'replays'/f'{rid}.json').read_text(encoding='utf-8')),rt().manifest)

    root=Path(__file__).resolve().parents[2]
    dist=root/'frontend'/'dist'
    if dist.is_dir():
        app.mount('/',StaticFiles(directory=dist,html=True),name='console')
    return app

app=create_app()
