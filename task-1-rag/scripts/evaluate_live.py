"""Small live evaluation using synthetic PDFs, real MiniLM, Chroma and Groq."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))
from streamlit.testing.v1 import AppTest
from test_runtime import pdf_bytes

def upload(name, text):
    data = pdf_bytes(text)
    return SimpleNamespace(name=name, getvalue=lambda: data)

app = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=60)
try:
    assert not app.exception, "Startup failed"
    fixtures = [
        upload("alpha-release.pdf", "Alpha release notes. The Orion launch is planned for March 4. The meeting venue is Cedar Room."),
        upload("beta-release.pdf", "Beta release notes. The Orion launch is planned for March 12. The meeting venue is Maple Room."),
    ]
    with patch("streamlit.file_uploader", side_effect=[fixtures, []]):
        app.run(timeout=60)
    assert not app.exception and not app.error, "Ingestion failed"
    cases = [
        ("alpha-release.pdf", "According to the selected release notes, when is the Orion launch planned?", "March 4"),
        ("beta-release.pdf", "According to the selected release notes, when is the Orion launch planned?", "March 12"),
        ("beta-release.pdf", "What is the Orion launch budget in the selected release notes?", None),
    ]
    for filename, question, expected in cases:
        for checkbox in app.checkbox:
            checkbox.set_value(checkbox.label == filename)
        app.run(timeout=30)
        app.chat_input[0].set_value(question).run(timeout=60)
        assert not app.exception, "App failed during a question"
        message = app.session_state["messages"][-1]
        assert not message.get("error"), message["content"]
        answer = message["content"]
        sources = message.get("sources", [])
        if expected:
            assert expected in " ".join(answer.split()).replace("**", ""), answer
            assert "[" in answer, answer
            assert sources and all(d.metadata["filename"] == filename for d in sources)
        else:
            assert answer == "I couldn't find that information in the uploaded documents.", answer
        print(json.dumps({"scope": filename, "question": question, "answer": answer,
                          "source_files": sorted({d.metadata["filename"] for d in sources}), "passed": True}), flush=True)
    print("LIVE_SMOKE: 3/3 passed", flush=True)
finally:
    index = app.session_state.filtered_state.get("_doctok_document_index")
    if index is not None:
        index.close()
