import csv

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image

# sqlalchemy
from sqlalchemy.orm import Session
from typing import List
import json

# import
import model
from core.dependencies import get_db
from core.schemas.museum_item import MuseumItem as MItem
from core.models.museum_item import MuseumItem
from v1.endpoints.reverse_search import functions as reverse_search_functions

museum_items_module = APIRouter()


@museum_items_module.get('/{item_id}', response_model=MItem)
async def create_similarity_request(item_id: str, db: Session=Depends(get_db)):
    item_id = ".".join(item_id.split(".")[:2])
    print(type(item_id))
    print("NOEN")
    print(item_id)
    item = db.query(MuseumItem).filter(MuseumItem.img_name == item_id).first()

    if not item:
        return JSONResponse(
            status_code=404,
            content={"message": "no such item"},
        )

    return item