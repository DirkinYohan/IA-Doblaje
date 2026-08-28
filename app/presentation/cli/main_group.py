"""CLI entrypoint principal basado en Typer.

T01: Implementacion MINIMA:
    - Comando ``--version`` (muestra fase + version)
    - Comando ``--help`` funcional
    - Sub-comandos placeholders (``analyze``, ``validate``, ``diagnose``)
      con mensaje informativo "implementado en siguientes tareas".

En Tareas sucesivas se ira completando cada comando con la logica real.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from app import __phase__, __status__, __title__, __version__
from app.core.constants import DeviceType


console = Console()


def _version_callback(value: bool) -> None:
    if value:
        _print_banner()
        raise typer.Exit(code=0)


app = typer.Typer(
    name="ia-doblaje",
    help="IA Profesional de Doblaje Automatico - Motor de Analisis de Audio (Fase 1)",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _print_banner() -> None:
    banner = (
        f"[bold cyan]{__title__}[/bold cyan]\n"
        f"[dim]Version:[/dim] [bold]{__version__}[/bold]  |  "
        f"[dim]Fase:[/dim] [yellow]{__phase__}[/yellow]  |  "
        f"[dim]Estado:[/dim] [magenta]{__status__}[/magenta]\n"
        f"[dim]Python:[/dim] {platform.python_version()}  |  "
        f"[dim]OS:[/dim] {platform.system()} {platform.release()}"
    )
    console.print(Panel(banner, border_style="cyan", expand=False))
    console.print()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        expose_value=False,
        is_flag=True,
        help="Mostrar version y salir.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        is_flag=True,
        help="Salida detallada.",
    ),
) -> None:
    """IA Doblaje Engine - Motor profesional de analisis de audio para doblaje.

    \b
    Estructura de comandos disponibles:
    - analyze   : Analizar un archivo multimedia (implementacion progresiva T04-T11)
    - validate  : Validar un JSON de analisis existente
    - diagnose  : Diagnostico del sistema (CPU/GPU/FFmpeg/PyTorch)
    """
    if ctx.invoked_subcommand is None:
        _print_banner()
        console.print(
            "[yellow]⚠️  FASE 1 EN DESARROLLO:[/yellow] "
            "el pipeline completo de analisis estara disponible al finalizar T11.\n"
        )
        console.print(
            "[dim]En T01 estan disponibles los comandos de diagnostico y placeholders.\n"
            "Use [bold]--help[/bold] en cualquier subcomando para ver opciones.[/dim]"
        )


@app.command("diagnose")
def diagnose_cmd(
    detailed: bool = typer.Option(
        False,
        "--detailed",
        is_flag=True,
        help="Mostrar informacion detallada del entorno.",
    ),
) -> None:
    """Diagnostico rapido del sistema: Python, PyTorch, GPU, FFmpeg, paths.

    T01: Muestra informacion basica del entorno y un resumen de salud.
    T02: Integrado con AppSettings reales + DeviceDetector + PathManager.
    """
    _print_banner()
    console.print(Panel.fit("[bold blue]DIAGNOSTICO DEL SISTEMA[/bold blue]", border_style="blue"))
    console.print()

    # --- Cargar AppSettings (si esta disponible, T02) ---
    settings = None
    try:
        from app.core.config import get_settings, clear_settings_cache
        clear_settings_cache()
        settings = get_settings()
    except Exception:  # noqa: BLE001 - defensivo
        settings = None

    # --- DeviceDetector (T02) ---
    dev_info = None
    try:
        from app.core.device import DeviceDetector
        preferred = DeviceType.AUTO if settings is None else settings.processing.device  # type: ignore[attr-defined]
        dd = DeviceDetector(
            allow_mps=getattr(getattr(settings, "gpu", None), "allow_mps", True),
            force_cpu=getattr(getattr(settings, "processing", None), "force_cpu", False),
        )
        dev_info = dd.detect(preferred=preferred)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        dev_info = None

    # --- Paths (T02) ---
    try:
        from app.core.paths import PathManager
        pm = PathManager.from_settings(settings)
        dirs = pm.ensure_dirs()
    except Exception:  # noqa: BLE001
        pm = None
        dirs = None

    console.print("[bold]Entorno Python:[/bold]")
    console.print(f"  • Interprete : [cyan]{sys.executable}[/cyan]")
    console.print(f"  • Version    : {platform.python_version()}")
    console.print(f"  • Platforma  : {platform.platform()}")
    cpu_logical = os.cpu_count() or 0
    cpu_physical = None
    ram_total_mb = 0
    ram_avail_mb = 0
    if dev_info is not None:
        cpu_logical = dev_info.cpu_cores_logical
        cpu_physical = dev_info.cpu_cores_physical
        ram_total_mb = dev_info.ram_total_mb
        ram_avail_mb = dev_info.ram_available_mb
    cores_str = str(cpu_logical)
    if cpu_physical:
        cores_str += f" (fisicos {cpu_physical})"
    console.print(f"  • CPU Cores  : {cores_str}")
    if ram_total_mb:
        console.print(
            f"  • RAM        : {ram_avail_mb/1024:.1f} GB libres / "
            f"{ram_total_mb/1024:.1f} GB totales"
        )
    console.print()

    # --- Configuracion del motor via AppSettings ---
    if settings is not None:
        try:
            processing = settings.processing
            gpu = settings.gpu
            safe_dump = settings.dump_safe()
            console.print("[bold]Configuracion (AppSettings):[/bold]")
            console.print(f"  • Profile       : [bold cyan]{processing.profile.value}[/bold cyan]")
            console.print(f"  • Device (req)  : {processing.device.value}")
            if dev_info is not None:
                console.print(
                    f"  • Device (det)  : [green]{dev_info.device_type.value}[/green] "
                    f"=> {dev_info.device_name}"
                )
            console.print(f"  • Auto-downgrade: {gpu.gpu_auto_downgrade_enabled}")
            console.print(f"  • VRAM headroom : {gpu.vram_headroom_mb} MB")
            console.print(
                f"  • Media max GB  : {settings.safety.max_media_size_gb:.1f}"
            )
            if detailed:
                hf = settings.hf
                hf_has = bool(getattr(hf.hf_token, "_secret_value", "")) if hasattr(hf, "hf_token") else False
                console.print(f"  • HF Token set  : {hf_has}")
        except Exception:  # noqa: BLE001
            pass
        console.print()

    # --- GPU / CUDA / VRAM ---
    try:
        import torch  # noqa: WPS433
        torch_ok = True
        torch_ver = torch.__version__
    except Exception as exc:  # noqa: BLE001
        torch_ok = False
        torch_ver = f"[red]NO DISPONIBLE: {exc!r}[/red]"

    console.print("[bold]IA / PyTorch:[/bold]")
    console.print(f"  • PyTorch instalado : [cyan]{torch_ok}[/cyan]")
    console.print(f"  • Version PyTorch   : {torch_ver}")

    if dev_info is not None:
        console.print(f"  • CUDA disponible   : {dev_info.cuda_available}")
        if dev_info.cuda_version:
            console.print(f"  • CUDA version      : {dev_info.cuda_version}")
        console.print(f"  • MPS disponible    : {dev_info.mps_available}")
        gpu = dev_info.primary_gpu
        if gpu is not None:
            console.print(f"  • GPU primaria      : [bold green]{gpu.name}[/bold green]")
            console.print(
                f"  • VRAM              : "
                f"{gpu.available_vram_gb:.1f} GB libres / "
                f"{gpu.total_vram_gb:.1f} GB totales "
                f"(usada {gpu.used_vram_gb:.1f} GB)"
            )
            # Politica Punto 2: sugerir perfiles compatibles
            try:
                avail = gpu.available_vram_gb
                from app.core.constants import QualityProfile
                fits = []
                for p, req in [
                    (QualityProfile.PERFORMANCE, 3.0),
                    (QualityProfile.BALANCED, 5.0),
                    (QualityProfile.QUALITY, 8.0),
                ]:
                    if req <= avail + 1e-6:
                        fits.append(p.value)
                if fits:
                    console.print(f"  • Perfiles VRAM OK  : [green]{', '.join(sorted(set(fits)))}[/green]")
                else:
                    console.print(
                        "  • Perfiles VRAM OK  : [yellow]ninguno[/yellow] "
                        "(requiere downgrade automatico o CPU)"
                    )
            except Exception:  # noqa: BLE001
                pass
        elif dev_info.device_type.is_accelerator is False and torch_ok:
            console.print("  • GPU detectada     : [dim]ninguna[/dim]")
        if dev_info.notes:
            for n in dev_info.notes:
                console.print(f"  • [dim]nota: {n}[/dim]")
    console.print()

    # --- Paths ---
    if dirs is not None and pm is not None:
        console.print("[bold]Directorios (PathManager):[/bold]")
        console.print(f"  • data/input     : {pm.data_input_dir}")
        console.print(f"  • data/output    : {pm.data_output_dir}")
        console.print(f"  • data/temporary : {pm.data_temp_dir}")
        console.print(f"  • models cache   : {pm.models_cache_dir}")
        console.print()

    if detailed:
        console.print(
            "[dim]Para chequeo completo de terceros (FFmpeg/FFprobe/HF/disco/VRAM/...) "
            "ejecutar:[/dim]\n"
            "    [bold]python scripts/verify_third_party.py[/bold]\n"
        )
    else:
        console.print(
            "[dim]Consejo: use --detailed para mas informacion. "
            "Para chequeo completo: python scripts/verify_third_party.py[/dim]"
        )


@app.command("analyze")
def analyze_cmd(
    input_path: str = typer.Argument(
        ...,
        help="Ruta al archivo multimedia de entrada (MP4/MKV/MOV/WAV/MP3/M4A).",
    ),
    profile: str = typer.Option(
        "balanced",
        "--profile",
        "-p",
        help="Perfil de procesamiento: quality | balanced | performance",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo de inferencia: auto | cuda | cpu | mps",
    ),
    output_dir: str = typer.Option(
        "data/output",
        "--output",
        "-o",
        help="Directorio de salida para los JSON de analisis.",
    ),
    job_id: Optional[str] = typer.Option(
        None,
        "--job-id",
        help="Identificador unico del trabajo (UUID autogenerado si no se provee).",
    ),
    force_save: bool = typer.Option(
        False,
        "--force-save",
        is_flag=True,
        help="Guardar resultados incluso si Quality Analysis reporta status=failed.",
    ),
) -> None:
    """Analizar un archivo multimedia y generar los JSON estructurados.

    Ejecuta el pipeline completo T01 → T14 (análisis de audio).
    """
    from app.application.pipeline.analyze_audio_pipeline import build_pipeline
    from app.application.pipeline.preflight import run_preflight

    _print_banner()

    # 1) Validar que el archivo existe
    from pathlib import Path as _Path

    input_p = _Path(input_path)
    if not input_p.exists():
        console.print(f"[red][FALLO][/red] Archivo no encontrado: {input_path}")
        raise typer.Exit(code=2)
    if not input_p.is_file():
        console.print(f"[red][FALLO][/red] No es un archivo regular: {input_path}")
        raise typer.Exit(code=2)

    # 2) Preflight (solo lectura)
    items = run_preflight()
    failed = [it for it in items if not it.ok]
    console.print(Panel.fit("[bold blue]PRECHECK DE PREPARACIÓN[/bold blue]", border_style="blue"))
    for it in items:
        mark = "[green]OK[/green]" if it.ok else "[red]FAIL[/red]"
        detail = f" — {it.detail}" if it.detail else ""
        console.print(f"  {mark}  {it.component}{detail}")
    if failed:
        console.print()
        console.print("[yellow]No se puede iniciar el pipeline: faltan dependencias/modelos.[/yellow]")
        raise typer.Exit(code=3)

    # 3) Construir y ejecutar el pipeline
    from app.core.config import get_settings

    settings = get_settings()
    try:
        pipeline = build_pipeline(settings=settings)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red][FALLO][/red] Error al construir el pipeline: {exc!r}")
        raise typer.Exit(code=4)

    console.print(Panel.fit("[bold magenta]EJECUCIÓN DEL PIPELINE T01→T14[/bold magenta]", border_style="magenta"))
    try:
        results = pipeline.run(
            str(input_p.resolve()),
            job_id=job_id,
            force_save=force_save,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red][FALLO][/red] Error durante el pipeline: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=5)

    console.print()
    console.print("[green][OK][/green] Pipeline finalizado correctamente.")
    console.print(f"  • Job ID   : {results.get('media_prep').job_id}")
    console.print(f"  • Output   : {settings.paths.data_output_dir / results.get('media_prep').job_id}")
    if results.get("output") is not None:
        out = results["output"]
        console.print(f"  • JSON     : {', '.join(out.file_sha256.keys())}")
    if results.get("cleanup") is not None:
        console.print(f"  • Cleanup  : cleaned={results['cleanup'].cleaned}")
    raise typer.Exit(code=0)


@app.command("validate")
def validate_cmd(
    analysis_json: str = typer.Argument(
        ...,
        help="Ruta al archivo analysis.json generado por el pipeline.",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        is_flag=True,
        help="Modo estricto (marca warnings como errores).",
    ),
) -> None:
    """Validar un JSON de analisis existente contra el schema.

    [yellow]⚠️  Implementacion completa en T08 junto a ValidationService.[/yellow]
    """
    _print_banner()
    console.print(f"[bold]Validar JSON:[/bold] {analysis_json}")
    console.print(f"[bold]Modo estricto:[/bold] {strict}")
    console.print()
    console.print(
        "[yellow]📌 Estado (T01):[/yellow] ValidationService Pydantic schema + Sanity checks "
        "se implementan en T08 de la Fase 1."
    )


if __name__ == "__main__":  # pragma: no cover
    app()
