from pydantic import BaseModel
from typing import List, Optional


# ---------------------------------
# Upload Response
# ---------------------------------

class TaskSubmitResponse(BaseModel):

    task_id: str

    status: str


# ---------------------------------
# Analysis Result
# ---------------------------------

class AnalysisResult(BaseModel):

    genre: str

    tempo: float

    valence: float

    arousal: float

    mood: str

    vibe: str

    explanations: List[str]


# ---------------------------------
# Task Status
# ---------------------------------

class TaskStatusResponse(BaseModel):

    task_id: str

    status: str

    result: Optional[AnalysisResult] = None
