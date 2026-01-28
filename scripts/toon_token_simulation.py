#!/usr/bin/env python3
"""
TOON Token Cost Simulation for znote-mcp

Compares token usage between current text format, JSON, and TOON
for high-fit endpoints identified in TOON_DESIGN.md
"""

import tiktoken

# Use cl100k_base (GPT-4/Claude tokenizer approximation)
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def print_comparison(name: str, formats: dict[str, str]):
    """Print side-by-side comparison with token counts."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    tokens = {k: count_tokens(v) for k, v in formats.items()}
    baseline = tokens.get("json", tokens.get("current"))

    for fmt, content in formats.items():
        t = tokens[fmt]
        if baseline and fmt != "json":
            pct = ((baseline - t) / baseline) * 100
            savings = f"({pct:+.1f}%)" if pct != 0 else ""
        else:
            savings = "(baseline)"
        print(f"\n--- {fmt.upper()} [{t} tokens] {savings} ---")
        print(content)

    print(f"\n{'─'*70}")
    print("TOKEN SUMMARY:")
    for fmt, t in sorted(tokens.items(), key=lambda x: x[1]):
        if baseline:
            pct = ((baseline - t) / baseline) * 100
            print(f"  {fmt:12} {t:5} tokens  ({pct:+.1f}% vs JSON)")
        else:
            print(f"  {fmt:12} {t:5} tokens")


# =============================================================================
# Simulation 1: FTS Search Results (20 results)
# =============================================================================

fts_json = '''{
  "query": "epistemology practice",
  "count": 20,
  "results": [
    {"id": "20240115T103045Z", "title": "Epistemology of Practice", "rank": 2.45, "snippet": "...the knowing-in-action reveals tacit dimensions..."},
    {"id": "20240112T091230Z", "title": "Tacit Knowledge", "rank": 2.12, "snippet": "...we know more than we can tell..."},
    {"id": "20240110T142015Z", "title": "Knowing-in-Action", "rank": 1.89, "snippet": "...reflection-in-action as epistemological stance..."},
    {"id": "20240108T083020Z", "title": "Professional Practice", "rank": 1.76, "snippet": "...artistry emerges from practice..."},
    {"id": "20240105T114530Z", "title": "Reflective Practitioner", "rank": 1.65, "snippet": "...surprise triggers reflection..."},
    {"id": "20240103T092145Z", "title": "Theory vs Practice", "rank": 1.54, "snippet": "...the gap between theory and practice..."},
    {"id": "20240101T160000Z", "title": "Embodied Knowledge", "rank": 1.43, "snippet": "...body as locus of knowing..."},
    {"id": "20231230T110015Z", "title": "Practical Wisdom", "rank": 1.32, "snippet": "...phronesis in professional contexts..."},
    {"id": "20231228T093045Z", "title": "Expert Intuition", "rank": 1.21, "snippet": "...pattern recognition in experts..."},
    {"id": "20231225T141520Z", "title": "Learning by Doing", "rank": 1.10, "snippet": "...Dewey's experiential learning..."},
    {"id": "20231222T085530Z", "title": "Craft Knowledge", "rank": 0.99, "snippet": "...apprenticeship and practice..."},
    {"id": "20231220T112000Z", "title": "Procedural Memory", "rank": 0.88, "snippet": "...skills become automatic..."},
    {"id": "20231218T094515Z", "title": "Practice Makes Perfect", "rank": 0.77, "snippet": "...deliberate practice research..."},
    {"id": "20231215T160045Z", "title": "Professional Judgment", "rank": 0.66, "snippet": "...judgment under uncertainty..."},
    {"id": "20231212T103020Z", "title": "Situated Cognition", "rank": 0.55, "snippet": "...knowledge is context-dependent..."},
    {"id": "20231210T091530Z", "title": "Communities of Practice", "rank": 0.44, "snippet": "...learning as participation..."},
    {"id": "20231208T142015Z", "title": "Skill Acquisition", "rank": 0.33, "snippet": "...Dreyfus model of expertise..."},
    {"id": "20231205T080030Z", "title": "Practical Reasoning", "rank": 0.22, "snippet": "...reasoning for action..."},
    {"id": "20231203T115545Z", "title": "Epistemic Virtues", "rank": 0.11, "snippet": "...intellectual humility in practice..."},
    {"id": "20231201T093000Z", "title": "Applied Epistemology", "rank": 0.05, "snippet": "...epistemology meets real world..."}
  ]
}'''

fts_current = '''Found 20 notes matching 'epistemology practice':

1. Epistemology of Practice (ID: 20240115T103045Z)
   Relevance: 2.45
   Match: ...the knowing-in-action reveals tacit dimensions...

2. Tacit Knowledge (ID: 20240112T091230Z)
   Relevance: 2.12
   Match: ...we know more than we can tell...

3. Knowing-in-Action (ID: 20240110T142015Z)
   Relevance: 1.89
   Match: ...reflection-in-action as epistemological stance...

4. Professional Practice (ID: 20240108T083020Z)
   Relevance: 1.76
   Match: ...artistry emerges from practice...

5. Reflective Practitioner (ID: 20240105T114530Z)
   Relevance: 1.65
   Match: ...surprise triggers reflection...

6. Theory vs Practice (ID: 20240103T092145Z)
   Relevance: 1.54
   Match: ...the gap between theory and practice...

7. Embodied Knowledge (ID: 20240101T160000Z)
   Relevance: 1.43
   Match: ...body as locus of knowing...

8. Practical Wisdom (ID: 20231230T110015Z)
   Relevance: 1.32
   Match: ...phronesis in professional contexts...

9. Expert Intuition (ID: 20231228T093045Z)
   Relevance: 1.21
   Match: ...pattern recognition in experts...

10. Learning by Doing (ID: 20231225T141520Z)
   Relevance: 1.10
   Match: ...Dewey's experiential learning...

11. Craft Knowledge (ID: 20231222T085530Z)
   Relevance: 0.99
   Match: ...apprenticeship and practice...

12. Procedural Memory (ID: 20231220T112000Z)
   Relevance: 0.88
   Match: ...skills become automatic...

13. Practice Makes Perfect (ID: 20231218T094515Z)
   Relevance: 0.77
   Match: ...deliberate practice research...

14. Professional Judgment (ID: 20231215T160045Z)
   Relevance: 0.66
   Match: ...judgment under uncertainty...

15. Situated Cognition (ID: 20231212T103020Z)
   Relevance: 0.55
   Match: ...knowledge is context-dependent...

16. Communities of Practice (ID: 20231210T091530Z)
   Relevance: 0.44
   Match: ...learning as participation...

17. Skill Acquisition (ID: 20231208T142015Z)
   Relevance: 0.33
   Match: ...Dreyfus model of expertise...

18. Practical Reasoning (ID: 20231205T080030Z)
   Relevance: 0.22
   Match: ...reasoning for action...

19. Epistemic Virtues (ID: 20231203T115545Z)
   Relevance: 0.11
   Match: ...intellectual humility in practice...

20. Applied Epistemology (ID: 20231201T093000Z)
   Relevance: 0.05
   Match: ...epistemology meets real world...'''

fts_toon = '''query: epistemology practice
results[20]{id,title,rank,snippet}:
  20240115T103045Z,Epistemology of Practice,2.45,...the knowing-in-action reveals tacit dimensions...
  20240112T091230Z,Tacit Knowledge,2.12,...we know more than we can tell...
  20240110T142015Z,Knowing-in-Action,1.89,...reflection-in-action as epistemological stance...
  20240108T083020Z,Professional Practice,1.76,...artistry emerges from practice...
  20240105T114530Z,Reflective Practitioner,1.65,...surprise triggers reflection...
  20240103T092145Z,Theory vs Practice,1.54,...the gap between theory and practice...
  20240101T160000Z,Embodied Knowledge,1.43,...body as locus of knowing...
  20231230T110015Z,Practical Wisdom,1.32,...phronesis in professional contexts...
  20231228T093045Z,Expert Intuition,1.21,...pattern recognition in experts...
  20231225T141520Z,Learning by Doing,1.10,...Dewey's experiential learning...
  20231222T085530Z,Craft Knowledge,0.99,...apprenticeship and practice...
  20231220T112000Z,Procedural Memory,0.88,...skills become automatic...
  20231218T094515Z,Practice Makes Perfect,0.77,...deliberate practice research...
  20231215T160045Z,Professional Judgment,0.66,...judgment under uncertainty...
  20231212T103020Z,Situated Cognition,0.55,...knowledge is context-dependent...
  20231210T091530Z,Communities of Practice,0.44,...learning as participation...
  20231208T142015Z,Skill Acquisition,0.33,...Dreyfus model of expertise...
  20231205T080030Z,Practical Reasoning,0.22,...reasoning for action...
  20231203T115545Z,Epistemic Virtues,0.11,...intellectual humility in practice...
  20231201T093000Z,Applied Epistemology,0.05,...epistemology meets real world...'''


# =============================================================================
# Simulation 2: List Notes (metadata only, 15 notes)
# =============================================================================

list_json = '''{
  "total": 42,
  "showing": 15,
  "offset": 0,
  "notes": [
    {"id": "20240120T143025Z", "title": "Complexity Theory Introduction", "note_type": "permanent", "project": "systems-thinking", "updated_at": "2024-01-20T14:30:25Z"},
    {"id": "20240119T091512Z", "title": "Emergent Properties", "note_type": "permanent", "project": "systems-thinking", "updated_at": "2024-01-19T09:15:12Z"},
    {"id": "20240118T162045Z", "title": "Feedback Loops", "note_type": "permanent", "project": "systems-thinking", "updated_at": "2024-01-18T16:20:45Z"},
    {"id": "20240117T103033Z", "title": "Nonlinear Dynamics", "note_type": "literature", "project": "systems-thinking", "updated_at": "2024-01-17T10:30:33Z"},
    {"id": "20240116T084520Z", "title": "Self-Organization", "note_type": "permanent", "project": "systems-thinking", "updated_at": "2024-01-16T08:45:20Z"},
    {"id": "20240115T142210Z", "title": "Attractors and Basins", "note_type": "permanent", "project": "systems-thinking", "updated_at": "2024-01-15T14:22:10Z"},
    {"id": "20240114T110545Z", "title": "Phase Transitions", "note_type": "literature", "project": "physics", "updated_at": "2024-01-14T11:05:45Z"},
    {"id": "20240113T093015Z", "title": "Chaos Theory Basics", "note_type": "permanent", "project": "systems-thinking", "updated_at": "2024-01-13T09:30:15Z"},
    {"id": "20240112T161230Z", "title": "Butterfly Effect", "note_type": "fleeting", "project": "systems-thinking", "updated_at": "2024-01-12T16:12:30Z"},
    {"id": "20240111T082045Z", "title": "Scale Invariance", "note_type": "permanent", "project": "physics", "updated_at": "2024-01-11T08:20:45Z"},
    {"id": "20240110T143520Z", "title": "Power Laws", "note_type": "permanent", "project": "statistics", "updated_at": "2024-01-10T14:35:20Z"},
    {"id": "20240109T091105Z", "title": "Network Effects", "note_type": "permanent", "project": "systems-thinking", "updated_at": "2024-01-09T09:11:05Z"},
    {"id": "20240108T164530Z", "title": "Small World Networks", "note_type": "literature", "project": "systems-thinking", "updated_at": "2024-01-08T16:45:30Z"},
    {"id": "20240107T102015Z", "title": "Preferential Attachment", "note_type": "permanent", "project": "systems-thinking", "updated_at": "2024-01-07T10:20:15Z"},
    {"id": "20240106T083545Z", "title": "Resilience Theory", "note_type": "permanent", "project": "systems-thinking", "updated_at": "2024-01-06T08:35:45Z"}
  ]
}'''

list_current = '''Notes (showing 15 of 42, offset 0):

1. Complexity Theory Introduction (ID: 20240120T143025Z)
   Type: permanent | Project: systems-thinking
   Updated: 2024-01-20 14:30:25

2. Emergent Properties (ID: 20240119T091512Z)
   Type: permanent | Project: systems-thinking
   Updated: 2024-01-19 09:15:12

3. Feedback Loops (ID: 20240118T162045Z)
   Type: permanent | Project: systems-thinking
   Updated: 2024-01-18 16:20:45

4. Nonlinear Dynamics (ID: 20240117T103033Z)
   Type: literature | Project: systems-thinking
   Updated: 2024-01-17 10:30:33

5. Self-Organization (ID: 20240116T084520Z)
   Type: permanent | Project: systems-thinking
   Updated: 2024-01-16 08:45:20

6. Attractors and Basins (ID: 20240115T142210Z)
   Type: permanent | Project: systems-thinking
   Updated: 2024-01-15 14:22:10

7. Phase Transitions (ID: 20240114T110545Z)
   Type: literature | Project: physics
   Updated: 2024-01-14 11:05:45

8. Chaos Theory Basics (ID: 20240113T093015Z)
   Type: permanent | Project: systems-thinking
   Updated: 2024-01-13 09:30:15

9. Butterfly Effect (ID: 20240112T161230Z)
   Type: fleeting | Project: systems-thinking
   Updated: 2024-01-12 16:12:30

10. Scale Invariance (ID: 20240111T082045Z)
    Type: permanent | Project: physics
    Updated: 2024-01-11 08:20:45

11. Power Laws (ID: 20240110T143520Z)
    Type: permanent | Project: statistics
    Updated: 2024-01-10 14:35:20

12. Network Effects (ID: 20240109T091105Z)
    Type: permanent | Project: systems-thinking
    Updated: 2024-01-09 09:11:05

13. Small World Networks (ID: 20240108T164530Z)
    Type: literature | Project: systems-thinking
    Updated: 2024-01-08 16:45:30

14. Preferential Attachment (ID: 20240107T102015Z)
    Type: permanent | Project: systems-thinking
    Updated: 2024-01-07 10:20:15

15. Resilience Theory (ID: 20240106T083545Z)
    Type: permanent | Project: systems-thinking
    Updated: 2024-01-06 08:35:45'''

list_toon = '''total: 42
showing: 15
offset: 0
notes[15]{id,title,note_type,project,updated_at}:
  20240120T143025Z,Complexity Theory Introduction,permanent,systems-thinking,2024-01-20T14:30:25Z
  20240119T091512Z,Emergent Properties,permanent,systems-thinking,2024-01-19T09:15:12Z
  20240118T162045Z,Feedback Loops,permanent,systems-thinking,2024-01-18T16:20:45Z
  20240117T103033Z,Nonlinear Dynamics,literature,systems-thinking,2024-01-17T10:30:33Z
  20240116T084520Z,Self-Organization,permanent,systems-thinking,2024-01-16T08:45:20Z
  20240115T142210Z,Attractors and Basins,permanent,systems-thinking,2024-01-15T14:22:10Z
  20240114T110545Z,Phase Transitions,literature,physics,2024-01-14T11:05:45Z
  20240113T093015Z,Chaos Theory Basics,permanent,systems-thinking,2024-01-13T09:30:15Z
  20240112T161230Z,Butterfly Effect,fleeting,systems-thinking,2024-01-12T16:12:30Z
  20240111T082045Z,Scale Invariance,permanent,physics,2024-01-11T08:20:45Z
  20240110T143520Z,Power Laws,permanent,statistics,2024-01-10T14:35:20Z
  20240109T091105Z,Network Effects,permanent,systems-thinking,2024-01-09T09:11:05Z
  20240108T164530Z,Small World Networks,literature,systems-thinking,2024-01-08T16:45:30Z
  20240107T102015Z,Preferential Attachment,permanent,systems-thinking,2024-01-07T10:20:15Z
  20240106T083545Z,Resilience Theory,permanent,systems-thinking,2024-01-06T08:35:45Z'''


# =============================================================================
# Simulation 3: Bulk Create Input (8 notes)
# =============================================================================

bulk_json = '''[
  {"title": "Introduction to Category Theory", "content": "Category theory is a branch of mathematics that deals with abstract structures and relationships between them.", "note_type": "literature", "project": "mathematics", "tags": ["category-theory", "mathematics", "abstraction"]},
  {"title": "Functors Explained", "content": "A functor is a mapping between categories that preserves structure.", "note_type": "permanent", "project": "mathematics", "tags": ["category-theory", "functors", "morphisms"]},
  {"title": "Natural Transformations", "content": "Natural transformations are morphisms between functors that respect the categorical structure.", "note_type": "permanent", "project": "mathematics", "tags": ["category-theory", "natural-transformations"]},
  {"title": "Monads in Programming", "content": "Monads are a design pattern from category theory used extensively in functional programming.", "note_type": "permanent", "project": "programming", "tags": ["monads", "functional-programming", "haskell"]},
  {"title": "Applicative Functors", "content": "Applicative functors are an intermediate abstraction between functors and monads.", "note_type": "permanent", "project": "programming", "tags": ["applicative", "functional-programming"]},
  {"title": "The Yoneda Lemma", "content": "The Yoneda lemma is a fundamental result in category theory about representable functors.", "note_type": "literature", "project": "mathematics", "tags": ["yoneda", "category-theory", "advanced"]},
  {"title": "Adjunctions", "content": "Adjunctions capture a fundamental relationship between functors going in opposite directions.", "note_type": "permanent", "project": "mathematics", "tags": ["adjunctions", "category-theory"]},
  {"title": "Limits and Colimits", "content": "Limits and colimits are universal constructions that generalize products, coproducts, and more.", "note_type": "permanent", "project": "mathematics", "tags": ["limits", "colimits", "category-theory"]}
]'''

bulk_toon = '''notes[8]{title,content,note_type,project,tags}:
  Introduction to Category Theory,"Category theory is a branch of mathematics that deals with abstract structures and relationships between them.",literature,mathematics,"category-theory,mathematics,abstraction"
  Functors Explained,A functor is a mapping between categories that preserves structure.,permanent,mathematics,"category-theory,functors,morphisms"
  Natural Transformations,"Natural transformations are morphisms between functors that respect the categorical structure.",permanent,mathematics,"category-theory,natural-transformations"
  Monads in Programming,Monads are a design pattern from category theory used extensively in functional programming.,permanent,programming,"monads,functional-programming,haskell"
  Applicative Functors,Applicative functors are an intermediate abstraction between functors and monads.,permanent,programming,"applicative,functional-programming"
  The Yoneda Lemma,The Yoneda lemma is a fundamental result in category theory about representable functors.,literature,mathematics,"yoneda,category-theory,advanced"
  Adjunctions,"Adjunctions capture a fundamental relationship between functors going in opposite directions.",permanent,mathematics,"adjunctions,category-theory"
  Limits and Colimits,"Limits and colimits are universal constructions that generalize products, coproducts, and more.",permanent,mathematics,"limits,colimits,category-theory"'''


# =============================================================================
# Run Simulations
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TOON TOKEN COST SIMULATION FOR ZNOTE-MCP")
    print("  Comparing: JSON | Current Text | TOON")
    print("="*70)

    print_comparison(
        "zk_fts_search (20 results)",
        {"json": fts_json, "current": fts_current, "toon": fts_toon}
    )

    print_comparison(
        "zk_list_notes (15 notes, metadata only)",
        {"json": list_json, "current": list_current, "toon": list_toon}
    )

    print_comparison(
        "zk_bulk_create_notes INPUT (8 notes)",
        {"json": bulk_json, "toon": bulk_toon}
    )

    # Summary
    print("\n" + "="*70)
    print("  AGGREGATE SUMMARY")
    print("="*70)

    all_json = count_tokens(fts_json + list_json + bulk_json)
    all_current = count_tokens(fts_current + list_current)  # bulk has no current
    all_toon = count_tokens(fts_toon + list_toon + bulk_toon)

    print(f"\nTotal JSON tokens:    {all_json}")
    print(f"Total TOON tokens:    {all_toon}")
    print(f"Savings vs JSON:      {all_json - all_toon} tokens ({((all_json - all_toon) / all_json) * 100:.1f}%)")

    print(f"\n(Current text format for search/list: {all_current} tokens)")
    fts_list_toon = count_tokens(fts_toon + list_toon)
    print(f"TOON for same endpoints:              {fts_list_toon} tokens")
    print(f"Savings vs current:                   {all_current - fts_list_toon} tokens ({((all_current - fts_list_toon) / all_current) * 100:.1f}%)")
