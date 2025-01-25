from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image

# sqlalchemy
from sqlalchemy.orm import Session
from typing import List, Annotated
import json

# import
import model
from core.dependencies import get_db
from core.schemas.museum_item import MuseumItem
from v1.endpoints.reverse_search import functions as reverse_search_functions

reverse_search_module = APIRouter()


# request similar images
@reverse_search_module.post('/', response_model=List[MuseumItem])
async def create_similarity_request(file: Annotated[UploadFile, File()], db: Session = Depends(get_db)):
    try:
        im = Image.open(file.file)
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        similar_images = [{k: float(v)} for k,v in model.main(im)]
    except Exception:
        raise HTTPException(status_code=500, detail="Couldn't process image")
    finally:
        file.file.close()
        im.close()
    return JSONResponse(
        status_code=200,
        content={"similar_images": similar_images},
    )