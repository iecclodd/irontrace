from pydantic import BaseModel, Field, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class NewSession(StrictModel):
    budget: float = Field(default=6, ge=0, le=1000)
    lambda_cost: float = Field(default=.02, ge=0, le=10)
    policy: str = 'information'
    costs: dict[str, float] | None = None

class Mutation(StrictModel):
    version: int = Field(ge=0)

class Acquire(Mutation):
    group_id: str
    idempotency_key: str = Field(min_length=8, max_length=128)

