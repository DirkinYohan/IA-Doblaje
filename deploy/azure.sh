#!/usr/bin/env bash
# Crea o actualiza el backend en Azure Container Apps (plan Consumption, escala a cero).
# Uso: deploy/azure.sh [etiqueta-de-imagen]
# Lee las claves del .env local; no se guardan en el repositorio.
set -euo pipefail

cd "$(dirname "$0")/.."

RG="${IA_AZ_RESOURCE_GROUP:-ia-doblaje-rg}"
LOCATION="${IA_AZ_LOCATION:-canadacentral}"
ENVIRONMENT="${IA_AZ_ENVIRONMENT:-ia-doblaje-env}"
APP="${IA_AZ_APP:-ia-doblaje-api}"
CPU="${IA_AZ_CPU:-4}"
MEMORY="${IA_AZ_MEMORY:-8Gi}"
IMAGE="ghcr.io/dirkinyohan/ia-doblaje-api:${1:-latest}"

value() {
  local line
  line=$(grep -E "^$1=" .env | tail -n 1 | tr -d '\r' || true)
  line="${line#*=}"
  case "$line" in
    \"*\") line="${line:1:${#line}-2}" ;;
    \'*\') line="${line:1:${#line}-2}" ;;
  esac
  echo "$line"
}

DATABASE_URL="$(value IA_DATABASE_URL)"
SUPABASE_URL="$(value SUPABASE_URL)"
SUPABASE_KEY="$(value SUPABASE_SERVICE_ROLE_KEY)"
SUPABASE_BUCKET="$(value SUPABASE_BUCKET)"
BREVO_KEY="$(value BREVO_API_KEY)"
MAIL_FROM="$(value MAIL_FROM_EMAIL)"
MAIL_NAME="$(value MAIL_FROM_NAME)"
PUBLIC_URL="${IA_PUBLIC_URL:-https://example.com}"

for name in DATABASE_URL SUPABASE_URL SUPABASE_KEY BREVO_KEY MAIL_FROM; do
  if [ -z "${!name}" ]; then
    echo "Falta $name en .env" >&2
    exit 1
  fi
done

# Sin una URL postgres válida el backend usaría SQLite dentro del contenedor,
# que se borra en cada reinicio: mejor no desplegar.
case "$DATABASE_URL" in
  postgres://*|postgresql://*) ;;
  *) echo "IA_DATABASE_URL no es una URL de PostgreSQL" >&2; exit 1 ;;
esac

az provider register --namespace Microsoft.App --wait >/dev/null
az group create --name "$RG" --location "$LOCATION" --output none

# IA_AZ_ENVIRONMENT_ID reutiliza un entorno existente (p. ej. si la suscripción solo admite uno).
if [ -n "${IA_AZ_ENVIRONMENT_ID:-}" ]; then
  ENVIRONMENT="$IA_AZ_ENVIRONMENT_ID"
elif ! az containerapp env show --name "$ENVIRONMENT" --resource-group "$RG" --output none 2>/dev/null; then
  az containerapp env create --name "$ENVIRONMENT" --resource-group "$RG" --location "$LOCATION" \
    --logs-destination none --output none
fi

SECRETS=(
  "database-url=$DATABASE_URL"
  "supabase-key=$SUPABASE_KEY"
  "brevo-key=$BREVO_KEY"
)
ENV_VARS=(
  "IA_DATABASE_URL=secretref:database-url"
  "SUPABASE_SERVICE_ROLE_KEY=secretref:supabase-key"
  "BREVO_API_KEY=secretref:brevo-key"
  "SUPABASE_URL=$SUPABASE_URL"
  "SUPABASE_BUCKET=${SUPABASE_BUCKET:-media}"
  "MAIL_FROM_EMAIL=$MAIL_FROM"
  "MAIL_FROM_NAME=${MAIL_NAME:-IA Doblaje}"
  "APP_PUBLIC_URL=$PUBLIC_URL"
  "CORS_ORIGINS=$PUBLIC_URL"
  "IA_DB_POOL_SIZE=3"
)
# URL pública del propio backend: el navegador descarga de aquí sin pasar por Vercel.
EXISTING_FQDN=$(az containerapp show --name "$APP" --resource-group "$RG"   --query properties.configuration.ingress.fqdn -o tsv 2>/dev/null || true)
if [ -n "$EXISTING_FQDN" ]; then
  ENV_VARS+=("IA_PUBLIC_API_URL=https://$EXISTING_FQDN")
fi

if az containerapp show --name "$APP" --resource-group "$RG" --output none 2>/dev/null; then
  az containerapp secret set --name "$APP" --resource-group "$RG" --secrets "${SECRETS[@]}" --output none
  az containerapp update --name "$APP" --resource-group "$RG" --image "$IMAGE" \
    --cpu "$CPU" --memory "$MEMORY" --min-replicas 0 --max-replicas 1 \
    --set-env-vars "${ENV_VARS[@]}" --output none
  # Los secretos nuevos solo se aplican al reiniciar la revisión.
  REVISION=$(az containerapp show --name "$APP" --resource-group "$RG" --query properties.latestRevisionName -o tsv)
  az containerapp revision restart --name "$APP" --resource-group "$RG" --revision "$REVISION" --output none
else
  az containerapp create --name "$APP" --resource-group "$RG" --environment "$ENVIRONMENT" \
    --image "$IMAGE" --cpu "$CPU" --memory "$MEMORY" --min-replicas 0 --max-replicas 1 \
    --ingress external --target-port 10000 --transport auto \
    --secrets "${SECRETS[@]}" --env-vars "${ENV_VARS[@]}" --output none
  NEW_FQDN=$(az containerapp show --name "$APP" --resource-group "$RG" --query properties.configuration.ingress.fqdn -o tsv)
  az containerapp update --name "$APP" --resource-group "$RG"     --set-env-vars "IA_PUBLIC_API_URL=https://$NEW_FQDN" --output none
fi

FQDN=$(az containerapp show --name "$APP" --resource-group "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo "Backend: https://$FQDN"
