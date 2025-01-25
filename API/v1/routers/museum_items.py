from fastapi import APIRouter
from v1.endpoints.museum_items.museum_items import museum_items_module

museum_items_router = APIRouter()

museum_items_router.include_router(
    museum_items_module,
    prefix="/items",
    tags=["museum_items"],
    responses={500: {"description": "Error occurred on server"}}
)