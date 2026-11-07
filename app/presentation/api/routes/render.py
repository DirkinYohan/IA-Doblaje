from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import Cookie
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import Response

from app.application.use_cases.build_subtitles import cue_from_dict
from app.presentation.api import db
from app.presentation.api.common import _own_project
from app.presentation.api.context import ApiContext
from app.presentation.api.schemas import RenderBody


def build(ctx: ApiContext) -> APIRouter:
    conn = ctx.conn
    paths = ctx.paths
    user_from_cookie = ctx.user_from_cookie
    router = APIRouter()

    def _render_track(project_id: str, user_id: str, lang: str) -> tuple[Any, list[Any], str]:
        project = _own_project(conn, project_id, user_id)
        row = db.one(conn, "SELECT * FROM subtitles WHERE project_id=? AND language=?", (project_id, lang))
        if row is None:
            raise HTTPException(status_code=404, detail="Subtítulos no encontrados")
        cues = [cue_from_dict(item) for item in json.loads(row["cues_json"])]
        if not cues:
            raise HTTPException(status_code=409, detail="La pista todavía no tiene subtítulos")
        return project, cues, str(row["updated_at"] or "")

    @router.post("/api/videos/{project_id}/render")
    def start_render(
        project_id: str, body: RenderBody | None = None, ia_session: str | None = Cookie(default=None)
    ) -> dict[str, Any]:
        from app.presentation.api import live
        from app.presentation.api import render

        row = user_from_cookie(ia_session)
        lang = (body.lang if body else "").strip().lower()
        if not lang:
            raise HTTPException(status_code=422, detail="Falta el idioma")
        project, cues, stamp = _render_track(project_id, row["id"], lang)
        return render.start(
            output_root=paths["output"], project=project, language=lang, cues=cues,
            stamp=stamp, source_of=live.media_source,
        )

    @router.get("/api/videos/{project_id}/render")
    def render_status(project_id: str, lang: str, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        from app.presentation.api import render

        row = user_from_cookie(ia_session)
        _project, _cues, stamp = _render_track(project_id, row["id"], lang)
        return render.status(paths["output"], project_id, lang, stamp)

    @router.get("/api/videos/{project_id}/render/file")
    def render_file(project_id: str, lang: str, ia_session: str | None = Cookie(default=None)) -> Response:
        from urllib.parse import quote

        from app.presentation.api import render

        row = user_from_cookie(ia_session)
        project, _cues, _stamp = _render_track(project_id, row["id"], lang)
        video = render.ready_file(paths["output"], project_id, lang)
        if video is None:
            raise HTTPException(status_code=404, detail="El vídeo todavía no está listo")
        base = Path(str(project["title"] or "video")).stem.replace("_", " ").strip() or "video"
        name = f"{base} ({lang}).mp4"
        ascii_name = name.encode("ascii", "ignore").decode() or f"video-{lang}.mp4"
        return FileResponse(
            video,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"
            },
        )

    return router
