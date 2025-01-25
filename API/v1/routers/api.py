from fastapi import APIRouter
from v1.routers.reverse_search import reverse_search_router
from v1.routers.museum_items import museum_items_router

router = APIRouter()

router.include_router(reverse_search_router)
router.include_router(museum_items_router)