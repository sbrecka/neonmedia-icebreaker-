import streamlit as st
import anthropic, os, requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

tools = [
    {
        "name": "fetch_website",
        "description": "Načte text z webové stránky firmy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL webu"}
            },
            "required": ["url"]
        }
    }
]

SYSTEM = """Jsi AI asistent pro cold email agenturu Neonmedia.
Tvojí jedinou úlohou je psát personalizované icebreakery pro české B2B firmy.

Pravidla:
- Vždy 1-2 věty, max 30 slov
- Piš v první osobě (já/my)
- Odkazuj na konkrétní detail z webu firmy — produkt, projekt, nebo číslo
- Nikdy nezačínaj "Dobrý den" ani "Ahoj"
- Pokud web neobsahuje užitečné info, napiš pouze: FALLBACK
- Žádné vysvětlování, žádné otázky — jen icebreaker

Příklad správného icebreakeru:
"Viděl jsem váš projekt pro Alza kde jste zkrátili onboarding o 40 % — zajímavý přístup k eliminaci friction v B2B."
Špatně: "Zajímalo by mě, jak děláte X?"
Správně: "Viděl jsem váš X — evidentně víte jak Y."
Vrať POUZE samotný icebreaker jako plain text. Žádný prefix, žádné uvozovky, žádné hvězdičky, žádné vysvětlení. Jediná výjimka: pokud nemáš info, napiš přesně: FALLBACK
"""



def fetch_website(url):
    try:
        response = requests.get(url, timeout=8)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text[:500]
    except Exception:
        return ""

def generate_icebreaker(firma, web):
    messages = [
        {"role": "user", "content": f"Vygeneruj personalizovaný icebreaker pro firmu {firma} (web: {web}). Piš česky, 1-2 věty."}
    ]
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        tools=tools,
        messages=messages,
        system=SYSTEM
    )
    if response.stop_reason == "tool_use":
        tool_block = response.content[-1]
        result = fetch_website(tool_block.input["url"])
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": result
            }]
        })
        final = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            tools=tools,
            messages=messages,
            system=SYSTEM
        )
        return final.content[0].text
    return response.content[0].text

st.title("Neonmedia — Icebreaker Generator")
st.caption("Auto-deploy z GitHubu ✓")

firma = st.text_input("Název firmy")
web = st.text_input("Web (URL)")

if st.button("Generovat"):
    if not firma or not web:
        st.warning("Vyplň obě pole.")
    else:
        with st.spinner("Generuji..."):
            icebreaker = generate_icebreaker(firma, web)
            st.success(icebreaker)

st.divider()
st.subheader("Batch — CSV")

uploaded = st.file_uploader("Nahraj CSV (musí mít sloupce 'name' a 'website')", type="csv")

if uploaded:
    import pandas as pd
    import io

    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.lower()

    if st.button("Generovat pro všechny"):
        results = []
        progress = st.progress(0)

        for i, row in df.iterrows():
            icebreaker = generate_icebreaker(row["name"], row["website"])
            results.append({**row.to_dict(), "icebreaker": icebreaker})
            progress.progress((i + 1) / len(df))

        result_df = pd.DataFrame(results)
        csv_out = result_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button("Stáhnout CSV", csv_out, "icebreakery.csv", "text/csv")
