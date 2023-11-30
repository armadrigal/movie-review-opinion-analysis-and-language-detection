from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def main():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Página HTML de ejemplo</title>
    </head>
    <body>
        <h1>¡Hola desde FastAPI!</h1>
        <p>Esta es una página HTML generada por FastAPI.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)