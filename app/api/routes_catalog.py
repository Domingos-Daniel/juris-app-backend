from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from app.services.pdf.ingestion import OFFICIAL_DOCUMENT_CATALOG
from app.services.rag.vector_store import legislation_vector_store

router = APIRouter(prefix="/catalog", tags=["Catalog"])

class CatalogItem(BaseModel):
    title: str
    scope: str
    status: str
    diploma_slug: str

class CatalogResponse(BaseModel):
    items: List[CatalogItem]

@router.get("", response_model=CatalogResponse)
async def get_catalog():
    indexed_slugs = legislation_vector_store.available_diploma_slugs()
    
    items = []
    for doc in OFFICIAL_DOCUMENT_CATALOG:
        slug = doc.get("diploma_slug")
        is_indexed = slug in indexed_slugs
        
        # Build scope from topics
        topics = doc.get("primary_topics", [])
        domains = doc.get("exclusive_domains", [])
        
        # Create a nice scope string
        all_scopes = topics + domains
        # Dedup keeping order
        seen = set()
        unique_scopes = []
        for s in all_scopes:
            formatted = s.replace("_", " ").capitalize()
            if formatted not in seen:
                seen.add(formatted)
                unique_scopes.append(formatted)
                
        scope_str = ", ".join(unique_scopes) + "." if unique_scopes else "Tema abrangente."
        
        status_str = "Validado no corpus" if is_indexed else "Ainda nao validado localmente"
        
        items.append(CatalogItem(
            title=doc.get("title", "Desconhecido"),
            scope=scope_str,
            status=status_str,
            diploma_slug=slug or ""
        ))
        
    return CatalogResponse(items=items)
