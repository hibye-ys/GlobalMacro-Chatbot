from datetime import date
from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class Property1(BaseModel):
    keyword: List[str] = Field(description="several keywords extracted from text")

    datetime: date = Field(description="Date the article was written in Headers")

    events: List[str] = Field(description="Major events that occurred")
