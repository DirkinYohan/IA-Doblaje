# Despliegue: backend en Render y frontend en Vercel

Son tres pasos y en total unos 20 minutos (la mayor parte es esperar la
primera compilación del backend).

| Parte | Repositorio | Servicio |
|---|---|---|
| Backend (API, Whisper, traducción) | `IA-Doblaje` | Render (Docker) |
| Frontend (Next.js) | `video_subtitles` | Vercel |
| Base de datos y vídeos | — | Supabase (el que ya usas) |
| Correos | — | Brevo (el que ya usas) |

## 1. Backend en Render

1. Entra en <https://dashboard.render.com> → **New** → **Blueprint**.
2. Conecta GitHub y elige el repositorio **IA-Doblaje**. Render detecta
   `render.yaml` y propone el servicio `ia-doblaje-api`.
3. Rellena las variables que te pide (los valores están en tu `.env` local):

   | Variable | Valor |
   |---|---|
   | `IA_DATABASE_URL` | la de tu `.env` (Supabase, pooler) |
   | `SUPABASE_URL` | la de tu `.env` |
   | `SUPABASE_SERVICE_ROLE_KEY` | la de tu `.env` |
   | `BREVO_API_KEY` | la de tu `.env` |
   | `MAIL_FROM_EMAIL` | el remitente verificado en Brevo |
   | `APP_PUBLIC_URL` | de momento `https://example.com`; se cambia en el paso 3 |
   | `CORS_ORIGINS` | de momento `https://example.com`; se cambia en el paso 3 |

4. Pulsa **Apply**. La primera compilación tarda 10-15 minutos: descarga
   Whisper y M2M100 y los deja dentro de la imagen.
5. Cuando termine, copia la URL del servicio (algo como
   `https://ia-doblaje-api.onrender.com`) y comprueba que
   `https://ia-doblaje-api.onrender.com/api/health` responde `{"status":"ok"}`.

**Plan:** `render.yaml` usa `1c-2g` (1 CPU, 2 GB). Con los modelos cargados la
API ocupa ~1 GB de RAM, así que el plan gratuito y el de 512 MB no sirven.
Si quieres que los subtítulos se generen más rápido, sube a `2c-4g`.

**Una sola instancia:** las sesiones en vivo, los tickets y los renders viven
en la memoria del proceso. No actives el escalado a varias instancias.

## 2. Frontend en Vercel

1. Entra en <https://vercel.com/new> e importa el repositorio
   **video_subtitles**. Vercel detecta Next.js solo.
2. En **Environment Variables** añade una sola:

   | Variable | Valor |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | la URL de Render del paso 1, sin `/` al final |

3. Pulsa **Deploy** y copia la URL que te da (algo como
   `https://video-subtitles.vercel.app`).

Con esa variable el frontend sabe todo lo que necesita:

- las páginas y la API normal pasan por Vercel (`/api/...`), así la cookie de
  sesión es del mismo dominio;
- el WebSocket de subtítulos en vivo va directo a `wss://` del backend;
- subir vídeos y descargar el MP4 con subtítulos van directos al backend con
  un ticket de un solo uso, porque Vercel corta las peticiones reenviadas a
  los 120 segundos.

## 3. Conectar los dos

En Render, abre `ia-doblaje-api` → **Environment** y cambia:

| Variable | Valor |
|---|---|
| `APP_PUBLIC_URL` | la URL de Vercel (para los enlaces de los correos) |
| `CORS_ORIGINS` | la URL de Vercel (para las subidas directas) |

Guarda; Render reinicia el servicio solo.

## Comprobar que funciona

1. Abre la URL de Vercel y crea una cuenta.
2. Te llega el correo de confirmación (mira también en spam); el enlace te
   lleva al inicio de sesión de Vercel.
3. Inicia sesión, sube un vídeo y dale a reproducir: los subtítulos aparecen
   mientras se generan.

## Actualizar

Cada `git push` a `main` vuelve a desplegar automáticamente en los dos
servicios.

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| La primera visita tarda ~1 minuto | El servicio arranca y carga los modelos | Normal tras un despliegue |
| Subir un vídeo da error de red | `CORS_ORIGINS` no tiene la URL exacta de Vercel | Cópiala con `https://` y sin `/` final |
| Los enlaces de los correos van a `example.com` | Falta el paso 3 | Cambia `APP_PUBLIC_URL` |
| No llegan los correos | Remitente sin verificar en Brevo | Brevo → Senders |
| El servicio se reinicia solo | Falta de memoria | Sube el plan a `2c-4g` |

## Probar la imagen en tu equipo (opcional)

```bash
docker build -t ia-doblaje-api .
docker run --rm -p 10000:10000 --env-file .env ia-doblaje-api
```

En Windows, si la carpeta está sincronizada con OneDrive, Docker no puede
leer los archivos que OneDrive tiene "sólo en la nube": marca la carpeta del
proyecto como "Mantener siempre en este dispositivo" antes de compilar.
