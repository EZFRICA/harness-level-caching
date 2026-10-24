import json, subprocess, sys, time, httpx
P = "inference-engineering-learn"
tok = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True).stdout.strip()
url = f"https://aiplatform.googleapis.com/v1/projects/{P}/locations/global/endpoints/openapi/chat/completions"
H = {"Authorization": f"Bearer {tok}", "x-goog-user-project": P}
system = "Tu es un assistant de voyage. " + " ".join(f"Note {i}: le voyageur prefere les trains de nuit et les musees le matin." for i in range(220))
for k, q in enumerate(["Plan de voyage d'ete a Paris, en une phrase.", "Et pour Lyon, en une phrase."]):
    t = time.time()
    r = httpx.post(url, headers=H, timeout=120, json={"model": sys.argv[1], "max_tokens": 60,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": q}]})
    ms = (time.time() - t) * 1000
    if r.status_code != 200:
        print(k, r.status_code, r.text[:300]); continue
    d = r.json(); u = d.get("usage", {})
    print(k, r.status_code, f"{ms:.0f} ms", "usage", json.dumps(u))
    print("   answer:", (d["choices"][0]["message"].get("content") or "")[:120].replace("\n", " "))
    time.sleep(2)
