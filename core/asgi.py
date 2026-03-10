import os
import django
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

django_application = get_asgi_application()

# Import FastAPI app
try:
    from fastapi_app import app as fastapi_application
except ImportError:
    fastapi_application = None

async def application(scope, receive, send):
    # Route all /api/socials/ paths to FastAPI for high performance
    if fastapi_application and scope['type'] == 'http' and scope['path'].startswith('/api/socials/'):
        print(f"⚡ [ASGI] Routing to FastAPI: {scope['path']}", flush=True)
        await fastapi_application(scope, receive, send)
    else:
        await django_application(scope, receive, send)
