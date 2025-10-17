from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pgeocode
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# --------------------
# Settings
# --------------------

class Settings(BaseSettings):
    npi_base_url: str = "https://npiregistry.cms.hhs.gov/api/"
    npi_api_version: str = "2.1"

    http_timeout_default: float = 10.0
    http_timeout_connect: float = 5.0
    http_timeout_read: float = 10.0
    http_timeout_write: float = 10.0

    data_dir: str = "data"

    class Config:
        env_prefix = "APP_"


settings = Settings()
DATA_DIR = Path(settings.data_dir)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# Models
# --------------------

class SearchRequest(BaseModel):
    zip: str = Field(..., min_length=5, max_length=10)
    state: Optional[str] = Field(default=None, min_length=2, max_length=2)
    specialty: str

    @field_validator("zip")
    @classmethod
    def normalize_zip(cls, v: str) -> str:
        return v.strip()

    @field_validator("specialty")
    @classmethod
    def normalize_specialty(cls, v: str) -> str:
        return v.strip().lower()


class CitySearchRequest(BaseModel):
    city: str
    state: str = Field(..., min_length=2, max_length=2)
    specialty: str

    @field_validator("city")
    @classmethod
    def normalize_city(cls, v: str) -> str:
        return v.strip()

    @field_validator("specialty")
    @classmethod
    def normalize_specialty(cls, v: str) -> str:
        return v.strip().lower()


class ProximityCityRequest(BaseModel):
    city: str
    state: str = Field(..., min_length=2, max_length=2)
    specialty: str
    origin_zip: str = Field(..., min_length=5, max_length=10)

    @field_validator("city")
    @classmethod
    def normalize_city(cls, v: str) -> str:
        return v.strip()

    @field_validator("specialty")
    @classmethod
    def normalize_specialty(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("origin_zip")
    @classmethod
    def normalize_zip(cls, v: str) -> str:
        return v.strip()


# --------------------
# Specialty → Taxonomy
# --------------------


SPECIALTY_TO_TAXONOMY: Dict[str, str] = {
    "internal_medicine": "207R00000X",
    "family_medicine": "207Q00000X",
    "emergency_medicine": "207P00000X",
    "pediatrics": "208000000X",
    "pediatrician": "208000000X",  # ADD THIS
    "obgyn": "207V00000X",
    "ob_gyn": "207V00000X",
    "cardiology": "207RC0000X",
    "cardiologist": "207RC0000X",  # ADD THIS - CRITICAL!
    "endocrinology": "207RE0101X",
    "gastroenterology": "207RG0100X",
    "geriatrics": "207RG0300X",
    "pulmonary": "207RP1001X",
    "psychiatry": "2084P0800X",
    "psychiatrist": "2084P0800X",  # ADD THIS
    "dermatology": "207N00000X",
    "dermatologist": "207N00000X",  # ADD THIS
    "neurology": "2084N0400X",
    "neurologist": "2084N0400X",  # ADD THIS
    "orthopedics": "207X00000X",
    "orthopedist": "207X00000X",  # ADD THIS
    "orthopaedic_surgery": "207X00000X",
    "urology": "208800000X",
    "primary_care": "363LP2300X",
}

# Near matches for primary-care–ish results when searching internal medicine
NEAR_PRIMARY_CARE: Dict[str, List[str]] = {
    "internal_medicine": [
        "207Q00000X",   # Family Medicine
        "363LP2300X",   # Nurse Practitioner, Primary Care
        "363A00000X",   # Physician Assistant
        "363AM0700X",   # Physician Assistant, Medical
    ],
    "family_medicine": [
        "207R00000X",
        "363LP2300X",
        "363A00000X",
        "363AM0700X",
    ],
}

# --------------------
# HTTP client
# --------------------

def _client() -> httpx.AsyncClient:
    timeout = httpx.Timeout(
        timeout=settings.http_timeout_default,
        connect=settings.http_timeout_connect,
        read=settings.http_timeout_read,
        write=settings.http_timeout_write,
    )
    return httpx.AsyncClient(timeout=timeout)

# --------------------
# Query assembly
# --------------------

# def _taxonomy_for_specialty(specialty: str) -> Tuple[str, List[str]]:
#     code = SPECIALTY_TO_TAXONOMY.get(specialty, SPECIALTY_TO_TAXONOMY.get("internal_medicine"))
#     # keywords used for fuzzy desc checks
#     keywords = {
#         "internal_medicine": ["internal", "primary", "primary care", "pcp", "general"],
#         "family_medicine": ["family", "primary", "primary care", "pcp", "general"],
#         "primary_care": ["primary", "primary care", "pcp", "general"],
#     }.get(specialty, [specialty.replace("_", " ")])
#     return code, keywords

def _taxonomy_for_specialty(specialty: str) -> Tuple[str, List[str]]:
    """
    Get taxonomy code and keywords for a specialty.
    Added debugging and better cardiology support.
    """
    specialty_lower = specialty.lower().strip()
    
    print(f"[TAXONOMY] Input specialty: '{specialty}' -> normalized: '{specialty_lower}'")
    
    # Check if it matches directly
    code = SPECIALTY_TO_TAXONOMY.get(specialty_lower)
    
    if not code:
        print(f"[TAXONOMY] No exact match found for '{specialty_lower}'")
        print(f"[TAXONOMY] Available specialties: {list(SPECIALTY_TO_TAXONOMY.keys())}")
        # Default fallback
        code = SPECIALTY_TO_TAXONOMY.get("internal_medicine")
        print(f"[TAXONOMY] Using fallback: internal_medicine -> {code}")
    else:
        print(f"[TAXONOMY] Found code: {code}")
    
    # Generate keywords
    keywords = {
        "internal_medicine": ["internal", "primary", "primary care", "pcp", "general"],
        "family_medicine": ["family", "primary", "primary care", "pcp", "general"],
        "primary_care": ["primary", "primary care", "pcp", "general"],
        "cardiologist": ["cardio", "heart", "cardiovascular"],
        "cardiology": ["cardio", "heart", "cardiovascular"],
        "pediatrician": ["pediatric", "child", "children"],
        "pediatrics": ["pediatric", "child", "children"],
    }.get(specialty_lower, [specialty_lower.replace("_", " ")])
    
    print(f"[TAXONOMY] Keywords: {keywords}")
    
    return code, keywords

def _base_params() -> Dict[str, str]:
    return {"version": settings.npi_api_version, "enumeration_type": "NPI-1", "limit": "50"}

async def query_by_zip(zip_code: str, state: Optional[str], specialty: str) -> Dict[str, Any]:
    taxonomy_code, _ = _taxonomy_for_specialty(specialty)
    params = _base_params() | {"postal_code": zip_code, "taxonomy": taxonomy_code}
    if state:
        params["state"] = state

    async with _client() as client:
        r = await client.get(settings.npi_base_url, params=params)
        r.raise_for_status()
        data = r.json()

    raw_path = DATA_DIR / f"npi_raw_{zip_code}_{taxonomy_code}.json"
    raw_path.write_text(json.dumps(data, indent=2))
    return data

async def query_by_city(city: str, state: str, specialty: str) -> Dict[str, Any]:
    taxonomy_code, _ = _taxonomy_for_specialty(specialty)
    params = _base_params() | {"city": city, "state": state, "taxonomy": taxonomy_code}

    async with _client() as client:
        r = await client.get(settings.npi_base_url, params=params)
        r.raise_for_status()
        data = r.json()

    raw_path = DATA_DIR / f"npi_raw_{city.lower().replace(' ', '_')}_{state}_{taxonomy_code}.json"
    raw_path.write_text(json.dumps(data, indent=2))
    return data

# --------------------
# Cleaning / scoring
# --------------------

def _best_location_address(addresses: List[Dict[str, Any]], want_state: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not addresses:
        return None
    locs = [a for a in addresses if (a.get("address_purpose") or "").upper() == "LOCATION"]
    if want_state:
        locs = [a for a in locs if (a.get("state") or "").upper() == want_state.upper()] or locs
    return locs[0] if locs else addresses[0]

def _score_taxonomy(
    taxonomies: List[Dict[str, Any]],
    requested_code: str,
    keywords: List[str],
    near_codes: Optional[List[str]] = None,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    if not taxonomies:
        return 0.0, None
    near = set((near_codes or []))
    req_prefix = requested_code[:4]
    best_s, best_t = 0.0, None

    for t in taxonomies:
        code = (t.get("code") or "").upper()
        desc = (t.get("desc") or "").lower()
        if code == requested_code:
            s = 1.0
        elif code in near:
            s = 0.8
        elif code.startswith(req_prefix):
            s = 0.7
        elif any(k in desc for k in keywords if k):
            s = 0.6
        else:
            s = 0.0
        if s > best_s:
            best_s, best_t = s, t

    return best_s, best_t

def _name_from_basic(basic: Dict[str, Any]) -> str:
    first = (basic.get("first_name") or "").strip()
    middle = (basic.get("middle_name") or "").strip()
    last = (basic.get("last_name") or "").strip()
    credential = (basic.get("credential") or "").strip()
    parts = [p for p in [first, middle, last] if p]
    name = " ".join(parts)
    if credential:
        name += f" {credential}"
    return name.strip() or None

def _extract_zip_from_addr(addr: Dict[str, Any]) -> Optional[str]:
    pc = (addr.get("postal_code") or "").strip()
    if not pc:
        return None
    return pc[:5]  # ZIP+4 → ZIP-5

def _to_public_card(
    entry: Dict[str, Any],
    matched_tax: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    basic = entry.get("basic") or {}
    addresses = entry.get("addresses") or []
    addr = _best_location_address(addresses)

    if not basic or not addr:
        return None

    line1 = addr.get("address_1")
    city = addr.get("city")
    state = addr.get("state")
    postal = addr.get("postal_code")
    phone = addr.get("telephone_number")

    # normalize ZIP-5 if ZIP+4
    zip5 = _extract_zip_from_addr(addr)
    if postal and len(postal) > 5:
        postal = postal[:5]

    specialty_label = (matched_tax.get("desc") if matched_tax else None) or "—"
    single_line = ", ".join(filter(None, [line1, city, state])) + (f" {postal}" if postal else "")

    return {
        "name": _name_from_basic(basic),
        "specialty": specialty_label,
        "phone": phone,
        "address": single_line.strip(),
        "zip": zip5,
    }


def clean_results_compact(
    raw: Dict[str, Any],
    specialty: str,
    scope_filter: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Clean and filter NPI search results.
    Fixed version with more lenient filtering and better debugging.
    """
    requested_code, keywords = _taxonomy_for_specialty(specialty)
    near = NEAR_PRIMARY_CARE.get(specialty, [])
    results = raw.get("results") or []
    
    if not results:
        print(f"[CLEAN] No results in raw data")
        return []

    print(f"[CLEAN] Processing {len(results)} raw results for specialty: {specialty}")
    print(f"[CLEAN] Requested taxonomy code: {requested_code}")
    print(f"[CLEAN] Keywords: {keywords}")

    want_state = (scope_filter or {}).get("state")
    want_zip = (scope_filter or {}).get("zip")
    want_city = (scope_filter or {}).get("city")

    cleaned: List[Dict[str, Any]] = []
    stats = {"no_address": 0, "location_filtered": 0, "low_score": 0, "added": 0}

    for idx, e in enumerate(results):
        addr = _best_location_address(e.get("addresses") or [], want_state=want_state)
        if not addr:
            stats["no_address"] += 1
            continue

        # Location filtering
        if want_zip:
            p = (addr.get("postal_code") or "")
            if p and not p.startswith(want_zip):
                c_ok = want_city and (addr.get("city") or "").upper() == want_city.upper()
                if not c_ok:
                    stats["location_filtered"] += 1
                    continue

        # Get all taxonomies for debugging
        taxonomies = e.get("taxonomies") or []
        
        # Get taxonomy score
        score, matched_tax = _score_taxonomy(
            taxonomies, requested_code, keywords, near_codes=near
        )
        
        # Debug: Show first few results in detail
        if idx < 3:
            print(f"[CLEAN] Result {idx} details:")
            print(f"  Name: {_name_from_basic(e.get('basic', {}))}")
            print(f"  Taxonomies: {[(t.get('code'), t.get('desc')) for t in taxonomies[:3]]}")
            print(f"  Score: {score}")
            print(f"  Matched: {matched_tax.get('desc') if matched_tax else 'None'}")
        
        # CRITICAL FIX: Only accept providers with a meaningful score
        # If score is 0, the provider doesn't match at all
        if score < 0.6:  # Increased threshold to be more selective
            stats["low_score"] += 1
            continue

        card = _to_public_card(e, matched_tax)
        if card:
            cleaned.append(card)
            stats["added"] += 1
        else:
            print(f"[CLEAN] Result {idx}: Failed to convert to card")

    print(f"[CLEAN] Stats: {stats}")
    print(f"[CLEAN] Returning {len(cleaned)} cleaned results")

    # Deduplicate by (name, address)
    seen = set()
    unique: List[Dict[str, Any]] = []
    for c in cleaned:
        key = (c.get("name"), c.get("address"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    print(f"[CLEAN] After deduplication: {len(unique)} unique results")
    
    # CRITICAL: If we got zero results after filtering, log why
    if len(unique) == 0:
        print(f"[CLEAN] WARNING: Zero results after filtering!")
        print(f"[CLEAN] Original result count: {len(results)}")
        print(f"[CLEAN] Requested specialty: {specialty}")
        print(f"[CLEAN] Requested taxonomy: {requested_code}")
        print(f"[CLEAN] Sample taxonomies from first result:")
        if results and results[0].get("taxonomies"):
            for t in results[0].get("taxonomies")[:3]:
                print(f"  - {t.get('code')}: {t.get('desc')}")
    
    return unique
# --------------------
# Proximity helpers
# --------------------

_nom_us = pgeocode.Nominatim("us")

def _zip_to_latlon(zip5: str) -> Optional[Tuple[float, float]]:
    if not zip5 or len(zip5) < 5:
        return None
    rec = _nom_us.query_postal_code(zip5[:5])
    # pgeocode returns pandas Series; lat/lon may be NaN
    if rec is None or str(rec.latitude) == "nan" or str(rec.longitude) == "nan":
        return None
    return float(rec.latitude), float(rec.longitude)

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088  # mean Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def clean_results_compact_with_distance(
    raw: Dict[str, Any],
    specialty: str,
    origin_zip: str,
    scope_filter: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Builds the compact list, computes distance from origin_zip (ZIP centroid),
    sorts by distance ascending, and returns only entries we can locate.
    """
    base = clean_results_compact(raw, specialty, scope_filter=scope_filter)
    origin = _zip_to_latlon(origin_zip)
    if not origin:
        # If the origin ZIP can't be geocoded, just return the base list
        return base

    o_lat, o_lon = origin
    enriched: List[Dict[str, Any]] = []
    for item in base:
        zip5 = item.get("zip")
        dest = _zip_to_latlon(zip5) if zip5 else None
        if not dest:
            # skip items we can't place reliably
            continue
        d_km = _haversine_km(o_lat, o_lon, dest[0], dest[1])
        d_mi = d_km * 0.621371
        enriched.append({
            "name": item["name"],
            "specialty": item["specialty"],
            "phone": item["phone"],
            "address": item["address"],
            "distance_miles": round(d_mi, 1),
        })

    enriched.sort(key=lambda x: x["distance_miles"])
    return enriched

# --------------------
# Save helpers (optional)
# --------------------

def save_clean_compact(data: List[Dict[str, Any]], tag: str) -> Path:
    path = DATA_DIR / f"npi_clean_compact_{tag}.json"
    path.write_text(json.dumps(data, indent=2))
    return path