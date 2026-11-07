from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from fastapi import Cookie
from fastapi import HTTPException
from fastapi.responses import Response

from app.application.use_cases.build_subtitles import cue_from_dict
from app.application.use_cases.build_subtitles import cue_to_dict
from app.application.use_cases.build_subtitles import zip_exports
from app.infrastructure.subtitles.exporters import to_ass
from app.infrastructure.subtitles.exporters import to_srt
from app.infrastructure.subtitles.exporters import to_vtt
from app.presentation.api import db
from app.presentation.api.common import OFFICIAL
from app.presentation.api.common import _job_is_running
from app.presentation.api.common import _now
from app.presentation.api.common import _own_project
from app.presentation.api.common import _track_view
from app.presentation.api.context import ApiContext
from app.presentation.api.schemas import SubtitlesBody


def build(ctx: ApiContext) -> APIRouter:
    conn = ctx.conn
    user_from_cookie = ctx.user_from_cookie
    router = APIRouter()

    @router.get("/api/subtitles/{project_id}/cues")
    def get_cues(
        project_id: str,
        lang: str = "",
        start_ms: int = -1,
        end_ms: int = -1,
        since_index: int = 0,
        ia_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        if not lang:
            raise HTTPException(status_code=422, detail="Falta el idioma")
        item = db.one(
            conn,
            "SELECT * FROM subtitles WHERE project_id=? AND language=?",
            (project_id, lang),
        )
        if item is None:
            return {"language": lang, "cues": [], "total": 0, "complete": False}
        try:
            cues = json.loads(item["cues_json"])
        except json.JSONDecodeError:
            cues = []
        total = len(cues)
        if since_index > 0:
            cues = [cue for cue in cues if int(cue.get("index", 0)) > since_index]
        if start_ms >= 0:
            cues = [cue for cue in cues if int(cue.get("end_ms", 0)) > start_ms]
        if end_ms >= 0:
            cues = [cue for cue in cues if int(cue.get("start_ms", 0)) < end_ms]
        job = db.one(
            conn,
            "SELECT status, last_cue_ms FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        return {
            "language": lang,
            "cues": cues,
            "total": total,
            "complete": bool(job is not None and job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}),
            "last_cue_ms": int(job["last_cue_ms"] or 0) if job is not None else 0,
        }

    @router.get("/api/subtitles/{project_id}")
    def get_subtitles(project_id: str, lang: str = "", ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        if lang:
            item = db.one(conn, "SELECT * FROM subtitles WHERE project_id=? AND language=?", (project_id, lang))
            rows = [item] if item else []
        else:
            rows = db.many(conn, "SELECT * FROM subtitles WHERE project_id=?", (project_id,))
        job = db.one(
            conn,
            "SELECT status FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        processing = job is not None and _job_is_running(str(job["status"]))
        return {
            "subtitles": [_track_view(project_id, r, processing=processing) for r in rows if r is not None]
        }

    @router.put("/api/subtitles/{project_id}")
    def put_subtitles(project_id: str, body: SubtitlesBody, ia_session: str | None = Cookie(default=None)) -> dict[str, int]:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        language = body.language
        if language not in OFFICIAL and language != "und":
            raise HTTPException(status_code=422, detail="Idioma inválido")
        try:
            cues = [cue_to_dict(cue_from_dict(item)) for item in body.cues]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Subtítulos inválidos") from exc
        db.run(
            conn,
            """
            INSERT INTO subtitles (project_id, language, cues_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id, language) DO UPDATE SET cues_json=excluded.cues_json, updated_at=excluded.updated_at
            """,
            (project_id, language, json.dumps(cues, ensure_ascii=False), _now()),
        )
        return {"count": len(cues)}

    def _export(project_id: str, user_id: str, lang: str, kind: str) -> Response:
        _own_project(conn, project_id, user_id)
        row = db.one(conn, "SELECT * FROM subtitles WHERE project_id=? AND language=?", (project_id, lang))
        if row is None:
            raise HTTPException(status_code=404, detail="Subtítulos no encontrados")
        cues = [cue_from_dict(item) for item in json.loads(row["cues_json"])]
        if kind == "srt":
            body, media = to_srt(cues), "application/x-subrip"
            filename = f"{project_id}.{lang}.srt"
        elif kind == "vtt":
            body, media = to_vtt(cues), "text/vtt"
            filename = f"{project_id}.{lang}.vtt"
        else:
            body, media = to_ass(cues, language=lang), "text/x-ssa"
            filename = f"{project_id}.{lang}.ass"
        return Response(content=body.encode("utf-8"), media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @router.get("/api/export/{project_id}/srt")
    def export_srt(project_id: str, lang: str, ia_session: str | None = Cookie(default=None)) -> Response:
        return _export(project_id, user_from_cookie(ia_session)["id"], lang, "srt")

    @router.get("/api/export/{project_id}/vtt")
    def export_vtt(project_id: str, lang: str, ia_session: str | None = Cookie(default=None)) -> Response:
        return _export(project_id, user_from_cookie(ia_session)["id"], lang, "vtt")

    @router.get("/api/export/{project_id}/ass")
    def export_ass(project_id: str, lang: str, ia_session: str | None = Cookie(default=None)) -> Response:
        return _export(project_id, user_from_cookie(ia_session)["id"], lang, "ass")

    @router.get("/api/export/{project_id}/zip")
    def export_zip(project_id: str, ia_session: str | None = Cookie(default=None)) -> Response:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        rows = db.many(conn, "SELECT * FROM subtitles WHERE project_id=?", (project_id,))
        files: dict[str, str] = {}
        for item in rows:
            cues = [cue_from_dict(c) for c in json.loads(item["cues_json"])]
            lang = item["language"]
            files[f"{lang}.srt"] = to_srt(cues)
            files[f"{lang}.vtt"] = to_vtt(cues)
            files[f"{lang}.ass"] = to_ass(cues, language=lang)
        if not files:
            raise HTTPException(status_code=404, detail="No hay subtítulos")
        return Response(content=zip_exports(files), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{project_id}.zip"'})

    return router
