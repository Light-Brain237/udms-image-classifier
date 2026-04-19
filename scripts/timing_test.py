"""Quick script to hit the classify endpoint and measure wall time."""
import requests
import time

url = "http://127.0.0.1:8000/api/v1/classify"
img_path = "data/processed/test/pothole_road/pothole_road_000030.jpg"

with open(img_path, "rb") as f:
    t0 = time.perf_counter()
    r = requests.post(url, files={"file": ("test.jpg", f, "image/jpeg")})
    t1 = time.perf_counter()

print(f"HTTP status: {r.status_code}")
print(f"Wall time:   {(t1 - t0) * 1000:.0f}ms")

d = r.json()
print(f"inference_time_ms: {d.get('inference_time_ms')}")
cat = d.get("prediction", {}).get("category")
print(f"category: {cat}")
