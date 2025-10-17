SPECIALTY_TO_TAXONOMY = {
    "internal_medicine": "207R00000X",
    "family_medicine": "207Q00000X",
    "pediatrics": "208000000X",
    "dermatology": "207N00000X",
    "cardiology": "207RC0000X",
    "cardiologist": "207RC0000X",     # Internal Medicine, Cardiovascular Disease
    "obgyn": "207V00000X",
    "endocrinology": "207RE0101X",
    "neurology": "2084N0400X",
}

def to_taxonomy(user_specialty: str) -> str | None:
    key = user_specialty.strip().lower().replace(" ", "_")
    return SPECIALTY_TO_TAXONOMY.get(key)