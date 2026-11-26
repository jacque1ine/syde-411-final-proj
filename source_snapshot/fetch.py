from pathlib import Path
import requests,json
p=Path(__file__).resolve().parent/'raw';p.mkdir(exist_ok=True)
api='https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/'
for package,needle,name in [('fire-incidents','Fire Incidents Data.csv','fires.csv'),('fire-station-locations','fire-station-locations - 4326.geojson','stations.geojson'),('neighbourhoods','Neighbourhoods - 4326.geojson','neighbourhoods.geojson'),('neighbourhood-profiles','neighbourhood-profiles-2021-158-model','population.xlsx')]:
 meta=requests.get(api+'package_show',params={'id':package},timeout=60).json()['result']
 (p/(package+'_metadata.json')).write_text(json.dumps(meta,indent=2))
 r=next(r for r in meta['resources'] if r['name']==needle)
 dest=p/name
 if not dest.exists():
  res=requests.get(r['url'],timeout=120);res.raise_for_status();dest.write_bytes(res.content)
 print(name,dest.stat().st_size)
