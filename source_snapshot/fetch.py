"""Download missing Toronto data files and refresh the source metadata."""

from pathlib import Path
import requests
import json

raw_dir = Path(__file__).resolve().parent / "raw"
raw_dir.mkdir(exist_ok=True)
api = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/"
for package, resource_name, name in [
    ("fire-incidents", "Fire Incidents Data.csv", "fires.csv"),
    (
        "fire-station-locations",
        "fire-station-locations - 4326.geojson",
        "stations.geojson",
    ),
    ("neighbourhoods", "Neighbourhoods - 4326.geojson", "neighbourhoods.geojson"),
    (
        "neighbourhood-profiles",
        "neighbourhood-profiles-2021-158-model",
        "population.xlsx",
    ),
]:
    metadata = requests.get(
        api + "package_show", params={"id": package}, timeout=60
    ).json()["result"]
    (raw_dir / (package + "_metadata.json")).write_text(json.dumps(metadata, indent=2))
    r = next(r for r in metadata["resources"] if r["name"] == resource_name)
    destination = raw_dir / name
    if not destination.exists():
        response = requests.get(r["url"], timeout=120)
        response.raise_for_status()
        destination.write_bytes(response.content)
    print(name, destination.stat().st_size)
