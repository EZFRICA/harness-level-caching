import json, subprocess, sys, time, httpx
P = "inference-engineering-learn"
tok = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True).stdout.strip()
m = sys.argv[1]
url = f"https://aiplatform.googleapis.com/v1/projects/{P}/locations/global/publishers/google/models/{m}:generateContent"
H = {"Authorization": f"Bearer {tok}", "x-goog-user-project": P}
system = "Tu es un assistant de voyage. " + " ".join(f"Note {i}: le voyageur prefere les trains de nuit et les musees le matin." for i in range(220))
for k, q in enumerate(["Plan de voyage d'ete a Paris, en une phrase.", "Et pour Lyon, en une phrase.", "Et pour Nantes, en une phrase."]):
    t = time.time()
    r = httpx.post(url, headers=H, timeout=120, json={"systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": q}]}], "generationConfig": {"maxOutputTokens": 60}})
    ms = (time.time() - t) * 1000
    if r.status_code != 200:
        print(k, r.status_code, r.text[:300]); continue
    u = r.json().get("usageMetadata", {})
    print(k, f"{ms:.0f} ms", "prompt", u.get("promptTokenCount"), "cached", u.get("cachedContentTokenCount", 0), "thoughts", u.get("thoughtsTokenCount", 0))
    time.sleep(3)
