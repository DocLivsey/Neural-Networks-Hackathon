from fastapi import APIRouter, Depends, HTTPException

# sqlalchemy
from sqlalchemy.orm import Session

# import
from core.dependencies import get_db
from core.schemas.museum_item import MuseumItem
from v1.endpoints.classification import functions as classification_functions

classification_module = APIRouter()

