"""Probe Gemini prompt caching: implicit (repeat identical prefix) and explicit (CachedContent)."""
import asyncio, os, sys, time
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
FACT = ("- Le bloc mémoire numéro {i} décrit une contrainte stable de l'utilisateur : il travaille le matin, "
        "préfère les réponses courtes, teste sur un Pixel 4 XL et voyage entre Nantes et Abidjan.\n")

def prefix(n_tokens):
    out, i = [], 0
    while len("".join(out)) < n_tokens * 4:
        out.append(FACT.format(i=i)); i += 1
    return "You are an assistant with memory.\n## MEMORY\n" + "".join(out)

async def implicit(model, size, reps=3):
    sysmsg = prefix(size)
    rows = []
    for k in range(reps):
        r = await client.aio.models.generate_content(
            model=model, contents=f"Question {k}: réponds juste OK.",
            config=types.GenerateContentConfig(system_instruction=sysmsg, temperature=0))
        u = r.usage_metadata
        rows.append((u.prompt_token_count, u.cached_content_token_count or 0))
        await asyncio.sleep(1)
    print(f"implicit {model:24} ~{size:>5} tok -> (prompt, cached) {rows}", flush=True)

async def explicit(model, size):
    try:
        c = await client.aio.caches.create(model=model, config=types.CreateCachedContentConfig(
            system_instruction=prefix(size), ttl="120s"))
    except Exception as e:
        print(f"explicit {model:24} ~{size:>5} tok -> CREATE FAILED: {str(e)[:220]}", flush=True); return
    try:
        r = await client.aio.models.generate_content(model=model, contents="Réponds juste OK.",
            config=types.GenerateContentConfig(cached_content=c.name, temperature=0))
        u = r.usage_metadata
        print(f"explicit {model:24} ~{size:>5} tok -> prompt={u.prompt_token_count} cached={u.cached_content_token_count}", flush=True)
    finally:
        await client.aio.caches.delete(name=c.name)

async def main():
    models = sys.argv[1:] or ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    for m in models:
        for size in (1500, 3000, 5000):
            await implicit(m, size)
        for size in (1500, 5000):
            await explicit(m, size)

asyncio.run(main())
