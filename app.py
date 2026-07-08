import streamlit as st
import anthropic, os, requests, json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path

st.set_page_config(page_title="Neonmedia — Icebreaker Generator", page_icon="⚡", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: radial-gradient(circle at 20% 0%, rgba(108,92,231,0.10), transparent 45%),
                radial-gradient(circle at 100% 30%, rgba(70,160,255,0.08), transparent 40%);
}

h1 {
    font-weight: 800 !important;
    letter-spacing: -1.5px;
    background: linear-gradient(90deg, #ffffff 0%, #b9aaff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.pipeline-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 999px;
    background: rgba(108,92,231,0.12);
    border: 1px solid rgba(108,92,231,0.35);
    color: #c9bfff; font-size: 13px; font-weight: 500;
    margin-bottom: 28px;
}

div[data-testid="stTextInput"] input {
    border-radius: 12px !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    padding: 10px 14px !important;
}
div[data-testid="stTextInput"] input:focus {
    border: 1px solid #6C5CE7 !important;
    box-shadow: 0 0 0 3px rgba(108,92,231,0.25) !important;
}

.stButton button {
    border-radius: 10px !important;
    background: linear-gradient(90deg, #6C5CE7, #46A0FF) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 8px 22px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(108,92,231,0.35);
}

div[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: rgba(255,255,255,0.02) !important;
}

div[data-testid="stAlertContainer"] {
    border-radius: 12px !important;
}

div[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

load_dotenv(Path(__file__).parent / ".env")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"

tools = [
    {
        "name": "fetch_website",
        "description": "Načte text z webové stránky firmy.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL webu"}},
            "required": ["url"]
        }
    }
]

def fetch_website(url):
    try:
        response = requests.get(url, timeout=8)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:1500]
    except Exception:
        return ""

def call_with_tool(system, user_msg):
    messages = [{"role": "user", "content": user_msg}]
    response = client.messages.create(model=MODEL, max_tokens=500, tools=tools, messages=messages, system=system)
    if response.stop_reason == "tool_use":
        tool_block = response.content[-1]
        result = fetch_website(tool_block.input["url"])
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_block.id, "content": result}
        ]})
        response = client.messages.create(model=MODEL, max_tokens=500, tools=tools, messages=messages, system=system)
    return response.content[0].text

RESEARCH_SYSTEM = """Jsi research agent pro cold email agenturu Neonmedia.
Najdi na webu firmy 3-5 konkrétních faktů použitelných pro icebreaker: produkty, projekty, čísla, klienty.
Vrať je jako odrážky, žádný úvod ani závěr. Pokud web nic užitečného neobsahuje, napiš přesně: FALLBACK
"""

COPYWRITER_SYSTEM = """Jsi copywriter agent pro cold email agenturu Neonmedia.
Na základě dodaných faktů o firmě napiš icebreaker.

Pravidla:
- Vždy 1-2 věty, max 30 slov
- Piš v první osobě (já/my)
- Odkazuj na konkrétní fakt z výzkumu — produkt, projekt, nebo číslo
- Nikdy nezačínaj "Dobrý den" ani "Ahoj"
- Žádné vysvětlování, žádné otázky — jen icebreaker
- Pokud dostaneš od review agenta feedback, uprav icebreaker podle něj

Vrať POUZE samotný icebreaker, žádné uvozovky ani prefix.
"""

REVIEW_SYSTEM = """Jsi review agent pro cold email agenturu Neonmedia. Kontroluješ icebreakery podle pravidel:
- max 30 slov, 1-2 věty
- obsahuje konkrétní fakt (číslo, produkt, projekt) — ne obecnou frázi
- nezačíná "Dobrý den" ani "Ahoj"
- nezní jako otázka ("Zajímalo by mě...")

Odpověz JEN ve formátu JSON: {"verdict": "APPROVED" nebo "REJECTED", "feedback": "..."}
Pokud APPROVED, feedback nech prázdný string.
"""

def research(firma, web):
    return call_with_tool(RESEARCH_SYSTEM, f"Prozkoumej firmu {firma}, web: {web}")

def write_icebreaker(firma, facts, feedback=None):
    prompt = f"Firma: {firma}\nFakta:\n{facts}"
    if feedback:
        prompt += f"\n\nFeedback od review agenta, uprav podle něj: {feedback}"
    response = client.messages.create(
        model=MODEL, max_tokens=300, system=COPYWRITER_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def review(icebreaker):
    response = client.messages.create(
        model=MODEL, max_tokens=300, system=REVIEW_SYSTEM,
        messages=[{"role": "user", "content": icebreaker}]
    )
    text = response.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj

def run_pipeline(firma, web, max_revisions=2):
    facts = research(firma, web)
    if facts.strip() == "FALLBACK":
        return None

    feedback = None
    icebreaker = None
    for _ in range(max_revisions + 1):
        icebreaker = write_icebreaker(firma, facts, feedback)
        verdict = review(icebreaker)
        if verdict["verdict"] == "APPROVED":
            return icebreaker
        feedback = verdict["feedback"]

    return icebreaker

st.title("Neonmedia")
st.markdown('<span class="pipeline-pill">⚡ Research → Copywriter → Review</span>', unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("Icebreaker Generator")
    firma = st.text_input("Název firmy", placeholder="např. Leftclick")
    web = st.text_input("Web (URL)", placeholder="https://...")
    generate = st.button("Generovat", use_container_width=False)

if generate:
    if not firma or not web:
        st.warning("Vyplň obě pole.")
    else:
        with st.spinner("Research agent hledá fakta..."):
            facts = research(firma, web)

        if facts.strip() == "FALLBACK":
            st.error("FALLBACK — web neobsahuje použitelná fakta.")
        else:
            with st.expander("🔍 Research agent — nalezená fakta"):
                st.text(facts)

            feedback = None
            icebreaker = None
            for attempt in range(3):
                with st.spinner(f"Copywriter agent píše (pokus {attempt + 1})..."):
                    icebreaker = write_icebreaker(firma, facts, feedback)
                with st.spinner("Review agent kontroluje..."):
                    verdict = review(icebreaker)

                icon = "✅" if verdict["verdict"] == "APPROVED" else "🔁"
                with st.expander(f"{icon} Pokus {attempt + 1}: {icebreaker}", expanded=(verdict["verdict"] != "APPROVED")):
                    st.write(f"**Verdikt:** {verdict['verdict']}")
                    if verdict["feedback"]:
                        st.write(f"**Feedback:** {verdict['feedback']}")

                if verdict["verdict"] == "APPROVED":
                    break
                feedback = verdict["feedback"]

            st.success(icebreaker)

st.write("")
with st.container(border=True):
    st.subheader("Batch — CSV")
    uploaded = st.file_uploader("Nahraj CSV (musí mít sloupce 'name' a 'website')", type="csv")

    if uploaded:
        import pandas as pd

        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.lower()

        if st.button("Generovat pro všechny"):
            results = []
            progress = st.progress(0)

            for i, row in df.iterrows():
                icebreaker = run_pipeline(row["name"], row["website"], max_revisions=1)
                results.append({**row.to_dict(), "icebreaker": icebreaker or "FALLBACK"})
                progress.progress((i + 1) / len(df))

            result_df = pd.DataFrame(results)
            csv_out = result_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button("Stáhnout CSV", csv_out, "icebreakery.csv", "text/csv")
