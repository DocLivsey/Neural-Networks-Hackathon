from pydantic import BaseModel


class MuseumItem(BaseModel):
    object_id: str
    name: str
    description: str
    group_: str
    img_name: str


