"""
rdf_schema.py  -  eICU RDF Ontology: namespaces, URI helpers, class/property definitions.

Provides:
  * Namespace constants (eicu:, data:, skos:, rdf:, rdfs:, owl:, xsd:)
  * URI builder helpers for every entity class
  * insert_ontology_triples(store) -> inserts OWL class + property definitions
"""

import hashlib, re
from pyoxigraph import NamedNode, Literal, Quad, DefaultGraph, Store

# == Namespace URIs ==========================================================
EICU   = "http://eicu.mit.edu/ontology/"
DATA   = "http://eicu.mit.edu/data/"
SKOS   = "http://www.w3.org/2004/02/skos/core#"
RDF    = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS   = "http://www.w3.org/2000/01/rdf-schema#"
OWL    = "http://www.w3.org/2002/07/owl#"
XSD    = "http://www.w3.org/2001/XMLSchema#"
ICD9   = "http://purl.bioontology.org/ontology/ICD9CM/"

# == Commonly-used NamedNode shortcuts =======================================
RDF_TYPE       = NamedNode(RDF + "type")
RDFS_LABEL     = NamedNode(RDFS + "label")
RDFS_COMMENT   = NamedNode(RDFS + "comment")
OWL_CLASS      = NamedNode(OWL + "Class")
OWL_DPROP      = NamedNode(OWL + "DatatypeProperty")
OWL_OPROP      = NamedNode(OWL + "ObjectProperty")

# == eICU class nodes ========================================================
CLASS_HOSPITAL     = NamedNode(EICU + "Hospital")
CLASS_PATIENT      = NamedNode(EICU + "Patient")
CLASS_DIAGNOSIS    = NamedNode(EICU + "Diagnosis")
CLASS_ORGAN_SYSTEM = NamedNode(EICU + "OrganSystem")
CLASS_RAW_DRUG     = NamedNode(EICU + "RawDrugName")
SKOS_CONCEPT       = NamedNode(SKOS + "Concept")

# == Data property predicates ================================================
EICU_BED_CATEGORY     = NamedNode(EICU + "bedCategory")
EICU_TEACHING_STATUS  = NamedNode(EICU + "teachingStatus")
EICU_REGION           = NamedNode(EICU + "region")

EICU_AGE              = NamedNode(EICU + "age")
EICU_GENDER           = NamedNode(EICU + "gender")
EICU_ETHNICITY        = NamedNode(EICU + "ethnicity")
EICU_ICU_MORTALITY      = NamedNode(EICU + "icuMortality")
EICU_HOSPITAL_MORTALITY = NamedNode(EICU + "hospitalMortality")
EICU_APACHE_SCORE     = NamedNode(EICU + "apacheScore")

EICU_DIAGNOSIS_STRING = NamedNode(EICU + "diagnosisString")
EICU_ICD9_CODE        = NamedNode(EICU + "icd9Code")
EICU_ORGAN_SYSTEM_NAME= NamedNode(EICU + "organSystemName")
EICU_CATEGORY         = NamedNode(EICU + "category")
EICU_PROBLEM          = NamedNode(EICU + "problem")
EICU_DETAIL           = NamedNode(EICU + "detail")
EICU_QUALIFIER        = NamedNode(EICU + "qualifier")
EICU_DIAGNOSIS_PRIORITY = NamedNode(EICU + "diagnosisPriority")

SKOS_PREF_LABEL   = NamedNode(SKOS + "prefLabel")
SKOS_ALT_LABEL    = NamedNode(SKOS + "altLabel")
SKOS_BROADER      = NamedNode(SKOS + "broader")
SKOS_EXACT_MATCH  = NamedNode(SKOS + "exactMatch")

# == Object property predicates (relationships) =============================
EICU_ORDERED          = NamedNode(EICU + "ordered")
EICU_ADMITTED_TO      = NamedNode(EICU + "admittedTo")
EICU_CONFIRMED_INFUSION = NamedNode(EICU + "confirmedInfusion")
EICU_HAS_DIAGNOSIS    = NamedNode(EICU + "hasDiagnosis")
EICU_BELONGS_TO       = NamedNode(EICU + "belongsTo")


# == URI builder helpers =====================================================

def _slug(text: str) -> str:
    """Normalise a free-text label into a URL-safe slug."""
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def hospital_uri(hospital_id: int) -> NamedNode:
    return NamedNode(f"{DATA}hospital/{hospital_id}")

def patient_uri(stayid: int) -> NamedNode:
    return NamedNode(f"{DATA}patient/{stayid}")

def diagnosis_uri(diagnosisstring: str, icd9code: str | None = None) -> NamedNode:
    """Deterministic URI from the diagnosis string + optional ICD-9 code."""
    raw = f"{diagnosisstring}|{icd9code or ''}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return NamedNode(f"{DATA}diagnosis/{h}")

def organ_system_uri(name: str) -> NamedNode:
    return NamedNode(f"{DATA}organ_system/{_slug(name)}")

def drug_uri(drugname: str) -> NamedNode:
    return NamedNode(f"{DATA}drug/{_slug(drugname)}")

def concept_uri(problem: str) -> NamedNode:
    return NamedNode(f"{DATA}concept/{_slug(problem)}")

def category_uri(category: str) -> NamedNode:
    return NamedNode(f"{DATA}category/{_slug(category)}")

def detail_uri(detail: str) -> NamedNode:
    return NamedNode(f"{DATA}detail/{_slug(detail)}")

def icd9_uri(code: str) -> NamedNode:
    # Clean code: remove dots for standard bioontology format if needed, 
    # but eICU codes usually don't have them or they are consistent.
    return NamedNode(f"{ICD9}{code}")


# == Typed literal helpers ===================================================

def xsd_string(val: str) -> Literal:
    return Literal(val, datatype=NamedNode(XSD + "string"))

def xsd_integer(val: int) -> Literal:
    return Literal(str(val), datatype=NamedNode(XSD + "integer"))

def xsd_float(val: float) -> Literal:
    return Literal(str(val), datatype=NamedNode(XSD + "float"))

def xsd_boolean(val: bool) -> Literal:
    return Literal("true" if val else "false", datatype=NamedNode(XSD + "boolean"))


# == Ontology bootstrap =====================================================

def insert_ontology_triples(store: Store) -> int:
    """Insert OWL class and property definitions into the store.  Returns triple count."""
    quads: list[Quad] = []
    dg = DefaultGraph()

    def _add(s, p, o):
        quads.append(Quad(s, p, o, dg))

    # --- Classes ---
    for cls, label in [
        (CLASS_HOSPITAL,     "Hospital"),
        (CLASS_PATIENT,      "Patient"),
        (CLASS_DIAGNOSIS,    "Diagnosis"),
        (CLASS_ORGAN_SYSTEM, "OrganSystem"),
        (CLASS_RAW_DRUG,     "RawDrugName"),
        (SKOS_CONCEPT,       "Clinical Concept (SKOS)"),
    ]:
        _add(cls, RDF_TYPE, OWL_CLASS)
        _add(cls, RDFS_LABEL, xsd_string(label))

    # --- Datatype properties ---
    data_props = [
        EICU_BED_CATEGORY, EICU_TEACHING_STATUS, EICU_REGION,
        EICU_AGE, EICU_GENDER, EICU_ETHNICITY,
        EICU_ICU_MORTALITY, EICU_HOSPITAL_MORTALITY, EICU_APACHE_SCORE,
        EICU_DIAGNOSIS_STRING, EICU_ICD9_CODE, EICU_ORGAN_SYSTEM_NAME,
        EICU_CATEGORY, EICU_PROBLEM, EICU_DETAIL, EICU_QUALIFIER,
        EICU_DIAGNOSIS_PRIORITY,
        SKOS_PREF_LABEL, SKOS_ALT_LABEL, SKOS_EXACT_MATCH,
    ]
    for dp in data_props:
        _add(dp, RDF_TYPE, OWL_DPROP)

    # --- Object properties ---
    obj_props = [
        EICU_ORDERED, EICU_ADMITTED_TO, EICU_CONFIRMED_INFUSION,
        EICU_HAS_DIAGNOSIS, EICU_BELONGS_TO, SKOS_BROADER,
    ]
    for op in obj_props:
        _add(op, RDF_TYPE, OWL_OPROP)

    # Bulk add
    for q in quads:
        store.add(q)

    return len(quads)


# == SPARQL prefix header (for queries) ======================================

SPARQL_PREFIXES = """
PREFIX eicu: <http://eicu.mit.edu/ontology/>
PREFIX data: <http://eicu.mit.edu/data/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
""".strip()
