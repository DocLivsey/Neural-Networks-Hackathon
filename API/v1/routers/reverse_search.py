from fastapi import APIRouter
from v1.endpoints.reverse_search.reverse_search import reverse_search_module

reverse_search_router = APIRouter()

reverse_search_router.include_router(
    reverse_search_module,
    prefix="/upload_file",
    tags=["reverse_search"],
    responses={500: {"description": "Error occurred on server"}}
)