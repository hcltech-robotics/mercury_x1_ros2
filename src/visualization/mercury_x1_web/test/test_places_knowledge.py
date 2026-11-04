"""named places (spoken name -> place) and the knowledge search used by voice chat"""
import os
import tempfile

from mercury_x1_web import places
from mercury_x1_web.knowledge import Knowledge, tokens

P = [{"name": "Kitchen", "x": 1, "y": 2, "yaw": 0, "aliases": ["coffee corner"]},
     {"name": "main entrance", "x": 0, "y": 0, "yaw": 0},
     {"name": "Demo table", "x": 3, "y": 1, "yaw": 1.5}]


def test_resolve_spoken_names():
    assert places.resolve("the kitchen", P)["name"] == "Kitchen"
    assert places.resolve("coffee corner", P)["name"] == "Kitchen"
    assert places.resolve("entrance", P)["name"] == "main entrance"
    assert places.resolve("the demo table area", P)["name"] == "Demo table"
    assert places.resolve("kitchn", P)["name"] == "Kitchen"
    assert places.resolve("parking lot", P) is None
    assert places.resolve("", P) is None


def test_upsert_and_file_roundtrip():
    out = places.upsert(P, "kitchen", 5, 6, 0.1)
    assert len(out) == 3 and [p for p in out if p["name"] == "kitchen"][0]["x"] == 5
    assert [p for p in out if p["name"] == "kitchen"][0]["aliases"] == ["coffee corner"]
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "m.places.json")
        places.save(f, out)
        assert places.load(f) == out
        assert places.load(os.path.join(d, "missing.json")) == []


def test_knowledge_search():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "a.md"), "w") as f:
            f.write("# Company facts as of Q2 FY27\n\n## Leadership\n\nThe CEO is Jane Doe.\n\n"
                    "## Results\n\nRevenue was 4 billion dollars in Q2 FY27.\n\nHeadcount was 220,000 people.\n\n"
                    "## Clients\n\nKey clients include Airbus, Siemens and Nokia.\n")
        k = Knowledge([d])
        assert k.search("who is the boss")[0]["text"] == "The CEO is Jane Doe."
        assert "Airbus" in k.search("do you work with Airbus?")[0]["text"]
        assert "220,000" in k.search("how many employees are there")[0]["text"]
        assert k.search("tell me a joke") == []
        assert k.search("revenue")[0]["source"] == "Company facts as of Q2 FY27"


def test_tokens():
    assert tokens("How many employees does HCL have?") == ["headcount", "hcltech"]
