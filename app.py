import streamlit as st
import anthropic, os, requests, json, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path

# TODO: až vymyslíš finální jméno appky, stačí přepsat tuhle proměnnou
APP_NAME = "Icebreaker"

st.set_page_config(page_title=APP_NAME, page_icon="◐", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background-color: #0a0a0b;
    background-image:
        radial-gradient(circle at 75% 15%, rgba(255,255,255,0.14), transparent 40%),
        radial-gradient(circle at 15% 85%, rgba(255,255,255,0.06), transparent 45%);
}

h1 {
    font-weight: 700 !important;
    letter-spacing: -2px;
    font-size: 3rem !important;
    color: #ffffff !important;
}

.pipeline-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: #b3b3b8; font-size: 13px; font-weight: 500;
    margin-bottom: 28px;
}

div[data-testid="stTextInput"] input {
    border-radius: 999px !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    padding: 10px 18px !important;
    color: #ffffff !important;
}
div[data-testid="stTextInput"] input:focus {
    border: 1px solid rgba(255,255,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.08) !important;
}

.stButton button {
    border-radius: 999px !important;
    background: #ffffff !important;
    color: #0a0a0b !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 8px 26px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(255,255,255,0.18);
    color: #0a0a0b !important;
}

div[data-testid="stExpander"] {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    background: rgba(255,255,255,0.02) !important;
}

div[data-testid="stAlertContainer"] {
    border-radius: 16px !important;
}

div[data-testid="stFileUploaderDropzone"] {
    border-radius: 16px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 20px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    background: rgba(255,255,255,0.02) !important;
}
</style>
""", unsafe_allow_html=True)

load_dotenv(Path(__file__).parent / ".env")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"

MEMORY_FILE = Path(__file__).parent / "agent_memory.json"
MAX_MISTAKES = 10  # kolik posledních review-feedbacků si agent pamatuje napříč běhy
MEMORY_LOCK = threading.Lock()  # batch běží paralelně (Den 30) — chrání soubor před souběžným zápisem

def load_memory():
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {"companies": {}, "mistakes": []}

def save_memory(memory):
    MEMORY_FILE.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")

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

RESEARCH_SYSTEM = """Jsi research agent pro cold email outreach.
Najdi na webu firmy 3-5 konkrétních faktů použitelných pro icebreaker: produkty, projekty, čísla, klienty.
Vrať je jako odrážky, žádný úvod ani závěr. Pokud web nic užitečného neobsahuje, napiš přesně: FALLBACK
"""

COPYWRITER_SYSTEM = """Jsi copywriter agent pro cold email outreach.
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

REVIEW_SYSTEM = """Jsi review agent pro cold email outreach. Kontroluješ icebreakery podle pravidel:
- max 30 slov, 1-2 věty
- obsahuje konkrétní fakt (číslo, produkt, projekt) — ne obecnou frázi
- nezačíná "Dobrý den" ani "Ahoj"
- nezní jako otázka ("Zajímalo by mě...")

Odpověz JEN ve formátu JSON: {"verdict": "APPROVED" nebo "REJECTED", "feedback": "..."}
Pokud APPROVED, feedback nech prázdný string.
"""

def research(firma, web):
    return call_with_tool(RESEARCH_SYSTEM, f"Prozkoumej firmu {firma}, web: {web}")

def write_icebreaker(firma, facts, feedback=None, known_mistakes=None):
    prompt = f"Firma: {firma}\nFakta:\n{facts}"
    if known_mistakes:
        prompt += "\n\nChyby, které jsi dělal v minulých bězích — nedělej je znovu:\n"
        prompt += "\n".join(f"- {m}" for m in known_mistakes)
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

def run_pipeline(firma, web, max_revisions=2, force=False):
    with MEMORY_LOCK:
        memory = load_memory()
        cached = memory["companies"].get(firma)
    if cached and not force:
        return cached["icebreaker"]

    facts = research(firma, web)
    if facts.strip() == "FALLBACK":
        return None

    feedback = None
    icebreaker = None
    for _ in range(max_revisions + 1):
        icebreaker = write_icebreaker(firma, facts, feedback, known_mistakes=memory["mistakes"])
        verdict = review(icebreaker)

        if verdict["verdict"] == "APPROVED":
            with MEMORY_LOCK:
                memory = load_memory()  # reload — jiné vlákno mohlo mezitím zapsat
                memory["companies"][firma] = {"icebreaker": icebreaker, "web": web, "date": date.today().isoformat()}
                save_memory(memory)
            return icebreaker

        feedback = verdict["feedback"]
        if feedback and feedback not in memory["mistakes"]:
            memory["mistakes"] = (memory["mistakes"] + [feedback])[-MAX_MISTAKES:]

    with MEMORY_LOCK:
        fresh = load_memory()
        fresh["mistakes"] = memory["mistakes"]
        save_memory(fresh)
    return icebreaker

st.title(APP_NAME)
st.markdown('<span class="pipeline-pill">◐ Research → Copywriter → Review</span>', unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("Vygeneruj icebreaker")
    firma = st.text_input("Název firmy", placeholder="např. Leftclick")
    web = st.text_input("Web (URL)", placeholder="https://...")
    generate = st.button("Generovat", use_container_width=False)

if generate:
    if not firma or not web:
        st.warning("Vyplň obě pole.")
    else:
        with MEMORY_LOCK:
            memory = load_memory()
            cached = memory["companies"].get(firma)

        if cached:
            st.info(f"◐ Z paměti (zpracováno {cached['date']}) — API se nevolalo")
            st.success(cached["icebreaker"])
        else:
            with st.spinner("Research agent hledá fakta..."):
                facts = research(firma, web)

            if facts.strip() == "FALLBACK":
                st.error("FALLBACK — web neobsahuje použitelná fakta.")
            else:
                with st.expander("🔍 Research agent — nalezená fakta"):
                    st.text(facts)
                if memory["mistakes"]:
                    with st.expander(f"◐ Paměť — {len(memory['mistakes'])} chyb z minulých běhů, které se copywriter snaží neopakovat"):
                        st.write("\n\n".join(memory["mistakes"]))

                feedback = None
                icebreaker = None
                for attempt in range(3):
                    with st.spinner(f"Copywriter agent píše (pokus {attempt + 1})..."):
                        icebreaker = write_icebreaker(firma, facts, feedback, known_mistakes=memory["mistakes"])
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
                    if feedback and feedback not in memory["mistakes"]:
                        memory["mistakes"] = (memory["mistakes"] + [feedback])[-MAX_MISTAKES:]

                with MEMORY_LOCK:
                    fresh = load_memory()
                    fresh["mistakes"] = memory["mistakes"]
                    if verdict["verdict"] == "APPROVED":
                        fresh["companies"][firma] = {"icebreaker": icebreaker, "web": web, "date": date.today().isoformat()}
                    save_memory(fresh)

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
            start = time.time()
            results = [None] * len(df)
            progress = st.progress(0)
            status = st.empty()
            done = 0

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(run_pipeline, row["name"], row["website"], 1): i
                    for i, row in df.iterrows()
                }
                for future in as_completed(futures):
                    i = futures[future]
                    icebreaker = future.result()
                    results[i] = {**df.iloc[i].to_dict(), "icebreaker": icebreaker or "FALLBACK"}
                    done += 1
                    progress.progress(done / len(df))
                    status.text(f"{done}/{len(df)} hotovo")

            elapsed = time.time() - start
            st.info(f"Hotovo za {elapsed:.1f}s (5 firem paralelně)")

            result_df = pd.DataFrame(results)
            csv_out = result_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button("Stáhnout CSV", csv_out, "icebreakery.csv", "text/csv")
