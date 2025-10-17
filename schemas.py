from pydantic import BaseModel, Field
from typing import Optional, List

class SearchRequest(BaseModel):
    zip: str = Field(..., min_length=5, max_length=10, description="ZIP or ZIP+4")
    specialty: str = Field(..., description="Human-friendly specialty label")

class ProviderBasic(BaseModel):
    npi: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization_name: Optional[str] = None
    primary_taxonomy_code: Optional[str] = None
    primary_taxonomy_desc: Optional[str] = None
    phone: Optional[str] = None
    address_1: Optional[str] = None
    address_2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None

class SearchResponse(BaseModel):
    result_count: int
    results: List[ProviderBasic]
    raw_saved_path: Optional[str] = None 

class CleanedProvider(BaseModel):
    npi: str = Field(..., description="NPI number")
    full_name: str
    taxonomy_code: str
    taxonomy_desc: str
    license: Optional[str] = None
    address_1: str
    city: str
    state: str
    postal_code: str  # 5-digit zip
    phone: Optional[str] = None