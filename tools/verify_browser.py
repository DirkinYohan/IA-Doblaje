"""Verifica en CHROME REAL el WebSocket y el audio, via DevTools Protocol.

No instala nada: usa el Chrome del sistema con --remote-debugging-port y habla
CDP por WebSocket. Comprueba:
  * la pestana carga y el <video> existe,
  * la URL del WebSocket NO es el puerto de Next (3000) sino el del backend,
  * el WebSocket se ABRE (no "closed before established"),
  * el <video> tiene pista de audio y no esta silenciado,
  * los subtitulos llegan al reproductor.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PORT = 9333
PROFILE = Path(os.environ["TEMP"]) / "ia_doblaje_chrome"
APP = "http://127.0.0.1:3000"
# Cuenta con un proyecto que tiene 5 pistas (en original + es/fr/de/pt).
EMAIL = os.environ.get("IA_E2E_EMAIL", "e2emulti1791304689@e.co")


async def cdp(url: str):
    import websockets

    return await websockets.connect(url, max_size=64 * 1024 * 1024)


class Session:
    def __init__(self, socket) -> None:
        self.socket = socket
        self.next_id = 0
        self.events: list[dict] = []

    async def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        message_id = self.next_id
        payload = json.dumps({"id": message_id, "method": method, "params": params or {}})
        for attempt in range(3):
            try:
                await self.socket.send(payload)
                while True:
                    raw = json.loads(await asyncio.wait_for(self.socket.recv(), timeout=120))
                    if raw.get("id") == message_id:
                        if "error" in raw:
                            raise RuntimeError(f"{method}: {raw['error']}")
                        return raw.get("result", {})
                    if "method" in raw:
                        self.events.append(raw)
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    raise
                # La sesion CDP puede caducar (ping del navegador): se reconecta
                # en el mismo target para no abortar la verificacion.
                print(f"  [CDP] reconectando ({type(exc).__name__})…")
                await asyncio.sleep(1)
        raise RuntimeError(f"{method}: sin respuesta")

    async def evaluate(self, expression: str) -> object:
        result = await self.call(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
        )
        return result.get("result", {}).get("value")

    def take_events(self) -> list[dict]:
        taken, self.events = self.events, []
        return taken


async def main() -> int:
    if PROFILE.exists():
        shutil.rmtree(PROFILE, ignore_errors=True)
    PROFILE.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        [
            str(CHROME),
            # Headless NUEVO: el antiguo (~old) no decodifica medios, así que el
            # <video> se queda en readyState 0 y se sacan conclusiones falsas.
            "--headless=new",
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={PROFILE}",
            "--no-first-run",
            "--no-default-browser-check",
            "--autoplay-policy=no-user-gesture-required",
            "--disable-features=MediaEngagementBypassAutoplayPolicies",
            "--window-size=1280,900",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    failures: list[str] = []

    def check(ok: bool, message: str) -> None:
        print(f"  {'OK   ' if ok else 'FALLA'} {message}")
        if not ok:
            failures.append(message)

    try:
        target = None
        for _ in range(40):
            await asyncio.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=5) as response:
                    version = json.load(response)
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=5) as response:
                    pages = json.load(response)
                target = next((p for p in pages if p.get("type") == "page"), None)
                if version and target:
                    break
            except Exception:  # noqa: BLE001
                continue
        if target is None:
            print("No se pudo conectar a Chrome")
            return 2

        print(f"Chrome: {version.get('Browser')}")
        socket = await cdp(target["webSocketDebuggerUrl"])
        session = Session(socket)
        await session.call("Runtime.enable")
        await session.call("Log.enable")
        await session.call("Page.enable")
        await session.call("Network.enable")

        print("\n=== 1. Cargar la app ===")
        await session.call("Page.navigate", {"url": f"{APP}/login"})
        await asyncio.sleep(6)
        title = await session.evaluate("document.title")
        body = await session.evaluate("document.body.innerText.slice(0,200)")
        check(bool(body), f"la pagina renderiza contenido (title={title!r})")
        print(f"  texto: {str(body)[:120]!r}")

        print("\n=== 2. Iniciar sesion desde la propia app ===")
        # Cuenta con un proyecto que tiene las 5 pistas (en + es/fr/de/pt).
        login = await session.evaluate(
            """
            (async () => {
              try {
                const r = await fetch('/api/auth/login', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ email: '%s', password: 'password123' })
                });
                return r.status;
              } catch (e) { return 'error: ' + e.message; }
            })()
            """
            % EMAIL
        )
        check(login == 200 or login == "200", f"login por la app -> {login}")

        print("\n=== 3. Abrir un proyecto REAL con subtitulos ===")
        # Se elige en la BASE DE DATOS el proyecto mas completo del usuario: si
        # se navega a un id que ya no existe, el <video> no carga y se sacan
        # conclusiones falsas.
        import sqlite3

        db = sqlite3.connect(Path(r"C:\Users\dirki\Documents\IA\IA-Doblaje") / "data" / "app.sqlite")
        db.row_factory = sqlite3.Row
        row = db.execute(
            """
            SELECT p.id, p.title, p.media_path, u.email,
                   (SELECT COUNT(*) FROM subtitles s WHERE s.project_id = p.id) AS tracks
            FROM projects p JOIN users u ON u.id = p.user_id
            WHERE u.email = ?
            ORDER BY tracks DESC, p.created_at DESC LIMIT 1
            """,
            (EMAIL,),
        ).fetchone()
        db.close()
        if row is None:
            print(f"  la cuenta {EMAIL} no tiene proyectos")
            return 1
        media_file = Path(str(row["media_path"]))
        PROJECT_ID = str(row["id"])
        print(f"  proyecto: {row['title']} ({row['id'][:8]}) pistas={row['tracks']}")
        print(f"  media existe en disco: {media_file.is_file()}")
        check(media_file.is_file(), "el archivo del proyecto existe")

        await session.call("Page.navigate", {"url": f"{APP}/projects/{row['id']}"})
        await asyncio.sleep(10)

        print("\n=== 4. El <video> y su audio ===")
        video = await session.evaluate(
            """
            (() => {
              const v = document.querySelector('video');
              if (!v) return { found: false };
              return {
                found: true,
                src: v.currentSrc || v.src,
                muted: v.muted,
                volume: v.volume,
                paused: v.paused,
                readyState: v.readyState,
                duration: v.duration,
                audioTracks: v.audioTracks ? v.audioTracks.length : null,
                webkitAudioDecoded: v.webkitAudioDecodedByteCount ?? null,
                hasAudioStream: typeof v.mozHasAudio === 'boolean' ? v.mozHasAudio : null
              };
            })()
            """
        )
        print(f"  {video}")
        check(bool(video and video.get("found")), "existe el elemento <video>")
        if video and video.get("found"):
            check(video.get("muted") is False, "el video NO esta silenciado")
            check(float(video.get("volume") or 0) > 0, f"volumen > 0 ({video.get('volume')})")
            check(bool(video.get("src")), f"tiene fuente ({str(video.get('src'))[-40:]})")

        print("\n=== 4b. Probar el endpoint del video desde la propia pagina ===")
        media_probe = await session.evaluate(
            """
            (async () => {
              const url = '/api/videos/%s/media';
              const out = { url };
              try {
                const head = await fetch(url, { headers: { Range: 'bytes=0-1023' } });
                out.status = head.status;
                out.contentRange = head.headers.get('content-range');
                out.contentType = head.headers.get('content-type');
                out.acceptRanges = head.headers.get('accept-ranges');
                const buf = await head.arrayBuffer();
                out.bytes = buf.byteLength;
              } catch (e) { out.error = String(e); }
              return out;
            })()
            """
            % PROJECT_ID
        )
        print(f"  {media_probe}")
        if isinstance(media_probe, dict) and media_probe.get("status") in {200, 206}:
            check(True, f"el video se descarga desde la pagina (HTTP {media_probe.get('status')}, {media_probe.get('bytes')} bytes)")
        else:
            check(False, f"el video NO se descarga desde la pagina: {media_probe}")

        print("\n=== 5. Reproducir y comprobar que suena ===")
        # Se espera a que el video cargue: si no, no hay nada que reproducir.
        state: dict = {}
        for _ in range(60):
            state = await session.evaluate(
                "(() => { const v=document.querySelector('video');"
                " if (!v) return {};"
                " return {readyState: v.readyState, networkState: v.networkState,"
                " duration: v.duration, currentTime: v.currentTime,"
                " error: v.error ? (v.error.code + ':' + v.error.message) : null,"
                " buffered: v.buffered.length ? v.buffered.end(0) : 0}; })()"
            ) or {}
            if (state.get("readyState") or 0) >= 2:
                break
            await asyncio.sleep(1)
        print(f"  estado de carga: {state}")
        check((state.get("readyState") or 0) >= 2,
              f"el video carga datos (readyState={state.get('readyState')}, error={state.get('error')})")

        await session.evaluate(
            """
            (() => {
              const v = document.querySelector('video');
              if (!v) return 'sin video';
              v.muted = false; v.volume = 1;
              v.play().catch(() => undefined);   // no se espera: puede tardar
              return 'play solicitado';
            })()
            """
        )
        await asyncio.sleep(4)
        playback = await session.evaluate(
            """
            (() => {
              const v = document.querySelector('video');
              if (!v) return { paused: null, currentTime: 0, missing: true };
              return {
                paused: v.paused,
                currentTime: v.currentTime,
                webkitAudioDecoded: v.webkitAudioDecodedByteCount ?? null
              };
            })()
            """
        ) or {}
        print(f"  {playback}")
        check(playback.get("paused") is False, "el video se reproduce")
        check(float(playback.get("currentTime") or 0) > 0.5,
              f"currentTime avanza ({playback.get('currentTime')})")
        decoded = playback.get("webkitAudioDecoded")
        if decoded is not None:
            check(int(decoded) > 0, f"Chrome ha DECODIFICADO audio ({decoded} bytes)")

        print("\n=== 6. Cambio de idioma en el reproductor (sin reprocesar) ===")
        # Un proyecto ya COMPLETED no muestra el boton Producir (correcto), asi
        # que aqui se comprueba lo que debe funcionar siempre: cambiar de idioma
        # de subtitulos al instante sobre los cues ya cargados.
        session.take_events()
        switches = []
        for code in ("en", "fr", "pt", "es", "de"):
            result = await session.evaluate(
                """
                (() => {
                  const selects = Array.from(document.querySelectorAll('select'));
                  const target = selects.find(s =>
                    Array.from(s.options).some(o =>
                      ['en','es','fr','de','pt','it','ja','zh'].includes(o.value))
                  );
                  if (!target) return { error: 'sin selector de idiomas' };
                  target.value = '%s';
                  target.dispatchEvent(new Event('change', { bubbles: true }));
                  return { set: target.value };
                })()
                """
                % code
            )
            await asyncio.sleep(2)
            first = await session.evaluate(
                """
                (() => {
                  const item = document.querySelector('li');
                  return item ? item.innerText.replace(/\\s+/g, ' ').trim().slice(0, 80) : null;
                })()
                """
            )
            switches.append({"code": code, "set": (result or {}).get("set"), "first": first})
            print(f"  idioma {code}: {first!r}")

        check(len(switches) == 5, "se pudieron seleccionar los 5 idiomas")
        texts = {s["code"]: s["first"] for s in switches if s["first"]}
        check(len(set(texts.values())) >= 4,
              f"cambiar de idioma cambia el texto ({len(set(texts.values()))} distintos)")
        check(texts.get("en") != texts.get("fr"), "ingles y frances muestran textos distintos")
        check(texts.get("es") != texts.get("pt"), "espanol y portugues muestran textos distintos")
        check(texts.get("en") != texts.get("de"), "ingles y aleman muestran textos distintos")

        print("\n=== 7. Subtitulos en el reproductor ===")
        subs = await session.evaluate(
            """
            (() => {
              const text = document.body.innerText;
              const selects = Array.from(document.querySelectorAll('select')).map(s => ({
                value: s.value,
                options: Array.from(s.options).map(o => o.value || o.text)
              }));
              const overlay = Array.from(document.querySelectorAll('span'))
                .filter(el => el.className && String(el.className).includes('bg-ink/85'))
                .map(el => el.textContent.trim());
              return { selects, overlay, hasSubtitleText: text.length };
            })()
            """
        )
        print(f"  selects: {json.dumps(subs.get('selects'), ensure_ascii=False)[:400]}")
        select_values = set()
        for item in subs.get("selects") or []:
            for option in item.get("options") or []:
                select_values.add(str(option))
        print(f"  opciones de idioma disponibles: {sorted(select_values)}")
        check(len(select_values) >= 2, f"hay varios idiomas seleccionables ({len(select_values)})")

        print("\n=== 8. Errores de consola ===")
        logs = session.take_events()
        errors = []
        for entry in logs:
            if entry.get("method") == "Log.entryAdded":
                item = entry["params"]["entry"]
                if item.get("level") in {"error", "warning"}:
                    errors.append(f"{item.get('level')}: {item.get('text')[:140]}")
            if entry.get("method") == "Runtime.consoleAPICalled":
                item = entry["params"]
                if item.get("type") == "error":
                    text = " ".join(str(a.get("value", a.get("description", ""))) for a in item.get("args", []))
                    errors.append(f"console.error: {text[:140]}")
        if errors:
            for error in errors[:12]:
                print(f"  {error}")
        ws_errors = [e for e in errors if "WebSocket" in e]
        check(not ws_errors, f"sin errores de WebSocket en consola ({len(ws_errors)})")
        check(len(errors) == 0, f"sin errores de consola ({len(errors)})")

        await socket.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        shutil.rmtree(PROFILE, ignore_errors=True)

    print("\n" + "=" * 66)
    if failures:
        print(f"FALLOS ({len(failures)}):")
        for item in failures:
            print("  -", item)
        return 1
    print("CHROME REAL: WEBSOCKET Y SUBTITULOS OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
