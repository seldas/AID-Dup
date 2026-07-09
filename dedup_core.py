"""
Core deduplication logic for the standalone Dedup Study app -- a direct port
of askMyFAERS backend/app/services/dedup_blocking.py. Pure Python, no AI / DB
/ IO dependencies.

Skill A (bucketing): tags each case with a configurable bucket label
(default: purely demographic "age_band | sex | country"). Only cases sharing
a bucket tag are ever compared.

Skill B (analysis): scores every pair within one bucket using
structured-field similarity (Tier 2), narrows large buckets' ambiguous pairs
with a TF-IDF narrative pre-filter (Tier 2.5), and leaves the survivors for
the Tier 3 narrative LLM comparison (pipeline.py).

Every tunable can be overridden per call via the versioned skill `config`
dict (see versioning.DEFAULT_CONFIG); the module constants are the defaults.
"""

import math
import re
from collections import defaultdict, Counter
from datetime import datetime

HIGH_CONFIDENCE_THRESHOLD = 0.85
AMBIGUOUS_THRESHOLD = 0.55
MAX_GROUP_SIZE = 60
NARRATIVE_PREFILTER_MIN_GROUP_SIZE = 10
NARRATIVE_SIMILARITY_THRESHOLD = 0.15

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class UnionFind:
    """Disjoint-set-union with path compression + union by rank."""

    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> list:
        out = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return list(out.values())


def _norm(value) -> str:
    return str(value).strip().lower() if value not in (None, "") else ""


def _to_set(val) -> set:
    if not val:
        return set()
    if isinstance(val, (list, set, tuple)):
        return set(str(v).strip().lower() for v in val if str(v).strip())
    if isinstance(val, str):
        return {val.strip().lower()}
    return set()


def _age_bucket(age, band_years: int = 5) -> str:
    """Fixed-width age band; '' (unknown) is its own bucket."""
    import math
    try:
        age_val = float(age)
        if math.isnan(age_val):
            return ""
    except (TypeError, ValueError):
        return ""
    band = max(1, int(band_years))
    return str(int(age_val) // band * band)


def compute_case_tag(case: dict, config: dict = None) -> str:
    """Skill A: the bucket tag for one case dict. Default fields are purely
    demographic (age band, sex, country); the versioned config can change the
    field set (any of age_band / sex / country / drug / event) and the
    age-band width. Missing fields fall back to "unknown"."""
    bucketing = (config or {}).get("bucketing") or {}
    fields = bucketing.get("fields") or ["age_band", "sex", "country"]
    band_years = bucketing.get("age_band_years") or 5

    values = {
        "age_band": lambda: _age_bucket(case.get("age"), band_years),
        "sex": lambda: _norm(case.get("sex")),
        "country": lambda: _norm(case.get("country")),
        "drug": lambda: _norm(case.get("drug")),
        "event": lambda: _norm(case.get("event")),
    }
    parts = [
        (values[f]() or "unknown")
        for f in ["age_band", "sex", "country", "drug", "event"]
        if f in fields
    ]
    return " | ".join(parts)


def run_tagging(cases: list, config: dict = None) -> dict:
    """Skill A orchestration for a whole series. AI-free. A tag that would
    only cover a single case is dropped (a bucket of one can never be
    analyzed). Returns {case_id: tag} for the non-singleton tags."""
    raw_tags = {c["case_id"]: compute_case_tag(c, config=config) for c in cases}
    tag_counts = Counter(raw_tags.values())
    return {cid: tag for cid, tag in raw_tags.items() if tag_counts[tag] > 1}


def split_oversized_group(members: list, config: dict = None) -> list:
    """Splits a too-large tag group further on report-date proximity so
    pairwise scoring never grows unbounded off a single heavy bucket."""
    max_size = ((config or {}).get("scoring") or {}).get("max_group_size") or MAX_GROUP_SIZE
    if len(members) <= max_size:
        return [members]
    ordered = sorted(members, key=lambda m: m.get("initial_date") or "")
    return [ordered[i:i + max_size] for i in range(0, len(ordered), max_size)]


def score_pair(a: dict, b: dict, config: dict = None) -> float:
    """Structured-field similarity in [0, 1]. Pure metadata, no LLM. Fields
    absent on either side simply don't contribute."""
    # Auto-merge if manufacturer control numbers match exactly
    mfr_a = _get_mfr_control_number(a)
    mfr_b = _get_mfr_control_number(b)
    if mfr_a and mfr_b:
        core_a = extract_mfr_core(mfr_a)
        core_b = extract_mfr_core(mfr_b)
        if core_a and core_b and core_a == core_b:
            return 1.0

    scoring = (config or {}).get("scoring") or {}
    ambiguous_threshold = scoring.get("ambiguous_threshold", AMBIGUOUS_THRESHOLD)

    # Auto-route to AI if dates match exactly (meta or narrative) and they are compatible
    if check_date_match(a, b) and check_compatibility(a, b, config):
        return ambiguous_threshold + 0.05

    w = scoring.get("weights") or {}
    w_sex = w.get("sex", 1.0)
    w_country = w.get("country", 1.0)
    w_age = w.get("age", 1.0)
    w_drugs = w.get("drugs", 2.0)
    w_events = w.get("events", 2.0)
    w_date = w.get("date", 1.0)
    age_tolerance = scoring.get("age_tolerance_years") or 10
    date_tolerance = scoring.get("date_tolerance_days") or 30

    weights_total = 0.0
    score = 0.0

    def add(weight: float, match: float):
        nonlocal weights_total, score
        if weight <= 0:
            return  # a zero weight disables the field entirely
        weights_total += weight
        score += weight * match

    if _norm(a.get("sex")) and _norm(b.get("sex")):
        add(w_sex, 1.0 if _norm(a["sex"]) == _norm(b["sex"]) else 0.0)
    if _norm(a.get("country")) and _norm(b.get("country")):
        add(w_country, 1.0 if _norm(a["country"]) == _norm(b["country"]) else 0.0)
    try:
        age_a = float(a.get("age"))
        age_b = float(b.get("age"))
        import math
        if not math.isnan(age_a) and not math.isnan(age_b):
            age_diff = abs(age_a - age_b)
            add(w_age, max(0.0, 1.0 - age_diff / age_tolerance))
    except (TypeError, ValueError):
        pass

    drugs_a, drugs_b = _to_set(a.get("drugs")), _to_set(b.get("drugs"))
    if drugs_a or drugs_b:
        union = drugs_a | drugs_b
        add(w_drugs, len(drugs_a & drugs_b) / len(union) if union else 0.0)
    events_a, events_b = _to_set(a.get("events")), _to_set(b.get("events"))
    if events_a or events_b:
        union = events_a | events_b
        add(w_events, len(events_a & events_b) / len(union) if union else 0.0)

    date_a, date_b = a.get("initial_date"), b.get("initial_date")
    if date_a and date_b:
        try:
            d1 = datetime.fromisoformat(str(date_a)[:10])
            d2 = datetime.fromisoformat(str(date_b)[:10])
            day_diff = abs((d1 - d2).days)
            add(w_date, max(0.0, 1.0 - day_diff / date_tolerance))
        except ValueError:
            pass

    return score / weights_total if weights_total else 0.0


def _tokenize(text: str) -> list:
    return _TOKEN_RE.findall((text or "").lower())


def _tfidf_vectors(documents: dict) -> dict:
    """documents: {key: free_text} -> {key: {term: tfidf_weight}}, smoothed."""
    tokenized = {k: _tokenize(t) for k, t in documents.items()}
    doc_freq = Counter()
    for tokens in tokenized.values():
        doc_freq.update(set(tokens))
    n_docs = len(tokenized)

    vectors = {}
    for key, tokens in tokenized.items():
        if not tokens:
            vectors[key] = {}
            continue
        term_freq = Counter(tokens)
        vectors[key] = {
            term: (count / len(tokens)) * math.log((1 + n_docs) / (1 + doc_freq[term]))
            for term, count in term_freq.items()
        }
    return vectors


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    if not vec_a or not vec_b:
        return 0.0
    numerator = sum(weight * vec_b[term] for term, weight in vec_a.items() if term in vec_b)
    if not numerator:
        return 0.0
    norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
    norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
    return numerator / (norm_a * norm_b) if norm_a and norm_b else 0.0


def filter_ambiguous_pairs_by_narrative(members: list, ambiguous_pairs: list, config: dict = None) -> list:
    """Tier 2.5: for buckets over min_group_size, narrows the ambiguous-pair
    list with TF-IDF cosine similarity over free-text narratives before any
    pair reaches the Tier 3 LLM call."""
    prefilter = (config or {}).get("narrative_prefilter") or {}
    min_group_size = prefilter.get("min_group_size", NARRATIVE_PREFILTER_MIN_GROUP_SIZE)
    similarity_threshold = prefilter.get("similarity_threshold", NARRATIVE_SIMILARITY_THRESHOLD)

    if len(members) <= min_group_size or not ambiguous_pairs:
        return ambiguous_pairs

    narratives = {m["case_id"]: m.get("narrative") or "" for m in members}
    vectors = _tfidf_vectors(narratives)

    return [
        (a, b, s) for (a, b, s) in ambiguous_pairs
        if _cosine_similarity(vectors.get(a, {}), vectors.get(b, {})) >= similarity_threshold
    ]


def score_group(members: list, config: dict = None) -> tuple:
    """Skill B Tier 2: scores every pair within one bucket's members. Returns
    (confirmed_pairs, ambiguous_pairs) of (case_id_a, case_id_b, score)."""
    scoring = (config or {}).get("scoring") or {}
    high_threshold = scoring.get("high_confidence_threshold", HIGH_CONFIDENCE_THRESHOLD)
    ambiguous_threshold = scoring.get("ambiguous_threshold", AMBIGUOUS_THRESHOLD)

    confirmed, ambiguous = [], []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            s = score_pair(a, b, config=config)
            if s >= high_threshold:
                confirmed.append((a["case_id"], b["case_id"], s))
            elif s >= ambiguous_threshold:
                ambiguous.append((a["case_id"], b["case_id"], s))
    return confirmed, ambiguous


def run_group_analysis(members: list, config: dict = None) -> dict:
    """Skill B Tiers 2 + 2.5 for one bucket's members. No LLM. Returns the
    union-find, confirmed/ambiguous pairs, and tier-funnel counters."""
    uf = UnionFind([m["case_id"] for m in members])
    confirmed_all = []
    ambiguous_seen = {}
    pairs_scored = 0

    for chunk in split_oversized_group(members, config=config):
        pairs_scored += len(chunk) * (len(chunk) - 1) // 2
        confirmed, ambiguous = score_group(chunk, config=config)
        for a, b, s in confirmed:
            uf.union(a, b)
            confirmed_all.append((a, b, s))
        for a, b, s in ambiguous:
            key = tuple(sorted((a, b)))
            if key not in ambiguous_seen or s > ambiguous_seen[key]:
                ambiguous_seen[key] = s

    ambiguous_pairs = [
        (a, b, s) for (a, b), s in ambiguous_seen.items() if uf.find(a) != uf.find(b)
    ]
    ambiguous_before_prefilter = len(ambiguous_pairs)
    ambiguous_pairs = filter_ambiguous_pairs_by_narrative(members, ambiguous_pairs, config=config)

    return {
        "union_find": uf,
        "confirmed_pairs": confirmed_all,
        "ambiguous_pairs": ambiguous_pairs,
        "pairs_scored": pairs_scored,
        "ambiguous_before_prefilter": ambiguous_before_prefilter,
    }


def average_linkage_clustering(nodes: list, pairwise_similarities: dict, threshold: float = 0.55) -> list:
    """Performs Hierarchical Agglomerative Clustering (HAC) on nodes using
    average linkage. Halt merges when the average similarity falls below
    threshold. pairwise_similarities: dict {(u, v): score} where u < v."""
    clusters = [[node] for node in nodes]

    def get_sim(u, v):
        if u == v:
            return 1.0
        key = tuple(sorted((u, v)))
        return pairwise_similarities.get(key, 0.0)

    def cluster_sim(c1, c2):
        total = 0.0
        for u in c1:
            for v in c2:
                total += get_sim(u, v)
        return total / (len(c1) * len(c2))

    while len(clusters) > 1:
        best_i, best_j = -1, -1
        best_sim = -1.0

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = cluster_sim(clusters[i], clusters[j])
                if sim > best_sim:
                    best_sim = sim
                    best_i, best_j = i, j

        if best_sim >= threshold:
            c_merged = clusters[best_i] + clusters[best_j]
            clusters.pop(best_j)
            clusters.pop(best_i)
            clusters.append(c_merged)
        else:
            break
    return clusters


def get_edge_density(nodes: list, similarities: dict, edge_threshold: float = 0.55) -> float:
    n = len(nodes)
    if n < 2:
        return 1.0
    e = 0
    for i in range(n):
        for j in range(i + 1, n):
            key = tuple(sorted((nodes[i], nodes[j])))
            if similarities.get(key, 0.0) >= edge_threshold:
                e += 1
    max_edges = n * (n - 1) // 2
    return e / max_edges if max_edges > 0 else 1.0


def get_required_density(group_size: int) -> float:
    if group_size < 50:
        return 0.7
    elif group_size <= 100:
        return 0.8
    elif group_size <= 200:
        return 0.9
    else:
        return 0.95


def split_group_by_density(nodes: list, similarities: dict, edge_threshold: float = 0.55) -> list:
    """Recursively splits a group using average linkage clustering (by raising thresholds)
    until all subgroups meet the required FDA JBI paper density threshold or can't be split further."""
    if len(nodes) < 2:
        return [nodes]
    density = get_edge_density(nodes, similarities, edge_threshold)
    required = get_required_density(len(nodes))
    if density >= required:
        return [nodes]
        
    # Split using average_linkage_clustering with a higher threshold
    sub_clusters = average_linkage_clustering(nodes, similarities, threshold=edge_threshold + 0.1)
    
    # If it didn't split into smaller clusters, force split by raising threshold even more
    if len(sub_clusters) == 1:
        sub_clusters = average_linkage_clustering(nodes, similarities, threshold=edge_threshold + 0.25)
        if len(sub_clusters) == 1:
            return [nodes]
            
    final_clusters = []
    for sub in sub_clusters:
        final_clusters.extend(split_group_by_density(sub, similarities, edge_threshold))
    return final_clusters


def _get_mfr_control_number(case: dict) -> str:
    raw = case.get("raw") or {}
    for k, v in raw.items():
        kl = str(k).lower().strip()
        if "mfr ctrl" in kl or "mfr control" in kl or "mfr #" in kl or "manufacturer control" in kl:
            if v and str(v).strip():
                return str(v).strip()
    return ""


def extract_mfr_core(mfr: str) -> str:
    if not mfr:
        return ""
    import re
    cleaned = re.sub(r'[\s/_:,;]+', '-', str(mfr).strip().upper())
    segments = [s.strip() for s in cleaned.split("-") if s.strip()]
    if not segments:
        return ""
    last = segments[-1]
    if len(last) >= 5:
        return last
    if len(segments) >= 2:
        combined = segments[-2] + last
        if len(combined) >= 5:
            return combined
    return last


def extract_dates_from_narrative(text: str) -> set:
    if not text:
        return set()
    import re
    from datetime import datetime
    
    dates = set()
    text = str(text).lower()
    
    # 1. Matches DD-MMM-YYYY or DD-MMM-YY (e.g. 13-dec-2018, 13-dec-18)
    months_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    pattern_mmm = r'\b(\d{1,2})-(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*-(\d{2,4})\b'
    for m in re.finditer(pattern_mmm, text):
        day = int(m.group(1))
        month_str = m.group(2)
        month = months_map[month_str]
        year_str = m.group(3)
        year = int(year_str)
        if len(year_str) == 2:
            year = 2000 + year if year < 50 else 1900 + year
        try:
            d = datetime(year, month, day)
            dates.add(d.strftime("%Y-%m-%d"))
        except ValueError:
            pass

    # 2. Matches YYYY-MM-DD
    pattern_iso = r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b'
    for m in re.finditer(pattern_iso, text):
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        try:
            d = datetime(year, month, day)
            dates.add(d.strftime("%Y-%m-%d"))
        except ValueError:
            pass

    # 3. Matches MM/DD/YYYY or DD/MM/YYYY
    pattern_slash = r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b'
    for m in re.finditer(pattern_slash, text):
        n1 = int(m.group(1))
        n2 = int(m.group(2))
        year_str = m.group(3)
        year = int(year_str)
        if len(year_str) == 2:
            year = 2000 + year if year < 50 else 1900 + year
            
        parsed = False
        if 1 <= n1 <= 12 and 1 <= n2 <= 31:
            try:
                d = datetime(year, n1, n2)
                dates.add(d.strftime("%Y-%m-%d"))
                parsed = True
            except ValueError:
                pass
        if not parsed and 1 <= n1 <= 31 and 1 <= n2 <= 12:
            try:
                d = datetime(year, n2, n1)
                dates.add(d.strftime("%Y-%m-%d"))
            except ValueError:
                pass

    return dates


def check_date_match(a: dict, b: dict) -> bool:
    # 1. Metadata date check
    date_a = a.get("initial_date")
    date_b = b.get("initial_date")
    if date_a and date_b:
        d1 = str(date_a)[:10]
        d2 = str(date_b)[:10]
        if d1 == d2:
            return True
            
    # 2. Narrative date check
    narr_a = a.get("narrative") or ""
    narr_b = b.get("narrative") or ""
    if narr_a and narr_b:
        dates_a = extract_dates_from_narrative(narr_a)
        dates_b = extract_dates_from_narrative(narr_b)
        if dates_a & dates_b:
            return True
            
    return False


def check_compatibility(a: dict, b: dict, config: dict = None) -> bool:
    scoring = (config or {}).get("scoring") or {}
    age_tolerance = scoring.get("age_tolerance_years") or 10
    
    # 1. Age compatibility
    try:
        age_a = float(a.get("age"))
        age_b = float(b.get("age"))
        import math
        if not math.isnan(age_a) and not math.isnan(age_b):
            if abs(age_a - age_b) > age_tolerance:
                return False
    except (TypeError, ValueError):
        pass
        
    # 2. Sex compatibility
    sex_a = _norm(a.get("sex"))
    sex_b = _norm(b.get("sex"))
    if sex_a and sex_b and sex_a != sex_b:
        return False
        
    # 3. Country compatibility
    country_a = _norm(a.get("country"))
    country_b = _norm(b.get("country"))
    if country_a and country_b and country_a != country_b:
        return False
        
    # 4. Drug compatibility
    drugs_a = _to_set(a.get("drugs"))
    drugs_b = _to_set(b.get("drugs"))
    if drugs_a and drugs_b:
        exact_share = bool(drugs_a & drugs_b)
        
        # Word-level overlap (words >= 4 chars)
        import re
        words_a = {w for d in drugs_a for w in re.split(r'\W+', d) if len(w) >= 4}
        words_b = {w for d in drugs_b for w in re.split(r'\W+', d) if len(w) >= 4}
        word_share = bool(words_a & words_b)
        
        # Brand synonym check: Lovenox <-> Enoxaparin
        str_a = " ".join(drugs_a)
        str_b = " ".join(drugs_b)
        syn_share = (("lovenox" in str_a and "enoxaparin" in str_b) or 
                     ("enoxaparin" in str_a and "lovenox" in str_b))
                     
        if not (exact_share or word_share or syn_share):
            return False
        
    return True


def run_smart_tagging(cases: list, config: dict = None) -> dict:
    """Skill A: Orchestrate smart bucketing. If type is not "smart", fall back to run_tagging."""
    bucketing_cfg = (config or {}).get("bucketing") or {}
    if bucketing_cfg.get("type") != "smart":
        return run_tagging(cases, config=config)

    # Smart bucketing is enabled. We will group cases using multi-key Union-Find.
    case_ids = [c["case_id"] for c in cases]
    uf = UnionFind(case_ids)

    # Gather matching keys for each case
    key_to_cases = defaultdict(list)
    
    # Check what smart matching features are enabled
    match_on_lot = bucketing_cfg.get("match_on_lot", True)
    match_on_serial = bucketing_cfg.get("match_on_serial", True)
    match_on_journal = bucketing_cfg.get("match_on_journal", True)
    match_on_mfr = bucketing_cfg.get("match_on_mfr", True)

    for c in cases:
        cid = c["case_id"]
        
        # 1. Demographic key (ignore generic tags with 'unknown')
        dem_tag = compute_case_tag(c, config)
        if dem_tag and "unknown" not in dem_tag:
            key_to_cases[f"demographic:{dem_tag}"].append(cid)
            
        # 2. Extract Manufacturer Control Number
        if match_on_mfr:
            mfr_val = _get_mfr_control_number(c)
            mfr_core = extract_mfr_core(mfr_val)
            if mfr_core:
                key_to_cases[f"mfr:{mfr_core}"].append(cid)
            
        # 2. Extract AI keys
        summary = c.get("ai_summary") or {}
        if not isinstance(summary, dict):
            summary = {}
            
        # Lot number keys
        if match_on_lot:
            lots = summary.get("lot_number")
            if lots:
                if isinstance(lots, list):
                    for lot in lots:
                        lot_norm = _norm(lot)
                        if lot_norm:
                            key_to_cases[f"lot:{lot_norm}"].append(cid)
                else:
                    lot_norm = _norm(lots)
                    if lot_norm:
                        key_to_cases[f"lot:{lot_norm}"].append(cid)

        # Serial number keys
        if match_on_serial:
            serials = summary.get("serial_number")
            if serials:
                if isinstance(serials, list):
                    for s in serials:
                        s_norm = _norm(s)
                        if s_norm:
                            key_to_cases[f"serial:{s_norm}"].append(cid)
                else:
                    s_norm = _norm(serials)
                    if s_norm:
                        key_to_cases[f"serial:{s_norm}"].append(cid)

        # Journal reference keys
        if match_on_journal:
            jr = summary.get("journal_reference")
            if jr:
                jr_norm = _norm(jr)
                if jr_norm:
                    key_to_cases[f"journal:{jr_norm}"].append(cid)

    # Union all cases sharing the same key
    for key, cids in key_to_cases.items():
        if len(cids) >= 2:
            first = cids[0]
            for other in cids[1:]:
                uf.union(first, other)

    # Name and return the tag mapping
    out = {}
    by_id = {c["case_id"]: c for c in cases}
    fallback_counter = 1
    
    for group in uf.groups():
        if len(group) < 2:
            continue
        
        # Determine tag name for this bucket
        # Prefer the first non-unknown demographic tag of the group members
        group_dem_tags = []
        for cid in group:
            dem = compute_case_tag(by_id[cid], config)
            if dem and "unknown" not in dem:
                group_dem_tags.append(dem)
        
        if group_dem_tags:
            tag_name = sorted(group_dem_tags)[0]
        else:
            # Look for shared lot/serial number to give it a descriptive name
            group_lots = []
            for cid in group:
                lots = (by_id[cid].get("ai_summary") or {}).get("lot_number")
                if lots:
                    if isinstance(lots, list):
                        group_lots.extend(lots)
                    else:
                        group_lots.append(lots)
            group_lots = [l for l in group_lots if l]
            if group_lots:
                tag_name = f"smart_group_lot_{_norm(group_lots[0])}"
            else:
                tag_name = f"smart_group_{fallback_counter}"
                fallback_counter += 1
                
        for cid in group:
            out[cid] = tag_name

    return out

