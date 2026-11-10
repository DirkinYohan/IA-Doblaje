# Despliegue: backend en Azure Container Apps y frontend en Vercel

| Parte | Dónde | Coste en reposo |
|---|---|---|
| Backend (API, Whisper, traducción) | Azure Container Apps, plan Consumption, escala a cero | 0 € |
| Imagen del backend | GitHub Container Registry (paquete público) | 0 € |
| Frontend (Next.js) | Vercel | 0 € (plan Hobby) |
| Base de datos | Azure Database for PostgreSQL, B1ms, 32 GB (Canada Central) | 0 € los 12 primeros meses (750 h/mes gratis); después ~13–15 USD/mes |
| Vídeos | Azure Blob Storage, contenedor privado `media` (East US) | 5 GB gratis 12 meses; después ~0,02 USD por GB al mes |
| Correos | Brevo | sin cambios |

## Cómo duerme y despierta

- Sin peticiones durante 5 minutos, el backend se apaga (0 réplicas) y no cobra.
- La siguiente petición lo despierta; el primer arranque tarda en descargar la
  imagen (~3,5 GB con los modelos) y cargarlos. Después responde normal.
- Un vídeo abierto mantiene el backend despierto (WebSocket de subtítulos).
- Si se duerme a mitad de un trabajo, al despertar vuelve a la cola y continúa
  desde lo guardado.

Recursos: 4 vCPU y 8 GiB, 1 réplica como máximo (con 2 vCPU los vídeos largos
se paraban). La concesión gratuita mensual de Azure (180 000 vCPU-s y
360 000 GiB-s, compartida por toda la suscripción) da para unas 12,5 horas
activas; después, unos 0,43 USD por hora despierto. Para volver a 2 vCPU:
`IA_AZ_CPU=2 IA_AZ_MEMORY=4Gi deploy/azure.sh`.

## 1. Imagen

Cada `push` a `main` que toque el backend ejecuta
`.github/workflows/image.yml`, que publica
`ghcr.io/dirkinyohan/ia-doblaje-api:<commit>` y `:latest`.

La primera vez el paquete queda privado: en GitHub → tu perfil → **Packages**
→ `ia-doblaje-api` → **Package settings** → **Change visibility** → Public.

## 2. Backend en Azure

Con la sesión de Azure iniciada (`az login`):

```bash
IA_PUBLIC_URL=https://<tu-app>.vercel.app deploy/azure.sh
```

El script crea (o actualiza) el grupo `ia-doblaje-rg`, el entorno sin Log
Analytics y la Container App, y guarda como secretos de Azure las claves que lee
de tu `.env`. Al final imprime la URL del backend.

Para desplegar una versión concreta: `deploy/azure.sh <commit>`.

Con dominio propio, `IA_PUBLIC_URL` es la dirección principal (la de los enlaces
de los correos) e `IA_EXTRA_ORIGINS` añade otras desde las que se puede usar la
API, separadas por comas:

```bash
IA_PUBLIC_URL=https://www.iadoblaje.online IA_EXTRA_ORIGINS=https://iadoblaje.online,https://video-subtitles-amm4.vercel.app deploy/azure.sh
```

Si la suscripción ya tiene un entorno de Container Apps (algunas solo admiten
uno), reutilízalo; la app sigue siendo independiente de las demás:

```bash
export MSYS_NO_PATHCONV=1   # solo en Git Bash de Windows
IA_AZ_ENVIRONMENT_ID=$(az containerapp env show -n <entorno> -g <grupo> --query id -o tsv) \
IA_PUBLIC_URL=https://<tu-app>.vercel.app deploy/azure.sh
```

### Base de datos y vídeos

- PostgreSQL acepta conexiones solo desde servicios de Azure y desde las IP
  añadidas en **Redes** del servidor (para desarrollar en local, añade la tuya).
  East US no admite PostgreSQL en esta suscripción; por eso está en Canada Central.
- El backend crea las tablas y activa RLS al arrancar.
- Con Azure Blob no hay límite de 50 MB por archivo: los vídeos se guardan sin
  comprimir y no hay tope de duración.
- En `.env`: `IA_DATABASE_URL`, `IA_STORAGE_BACKEND=azure`,
  `AZURE_STORAGE_CONNECTION_STRING` y `AZURE_STORAGE_CONTAINER`.

## 3. Frontend en Vercel

Importa el repositorio `video_subtitles` en Vercel y define una variable:

| Variable | Valor |
|---|---|
| `NEXT_PUBLIC_API_URL` | la URL que imprime `deploy/azure.sh` |

## Cambiar una clave

Edita `.env` y vuelve a ejecutar `deploy/azure.sh`: actualiza los secretos y crea
una revisión nueva.
