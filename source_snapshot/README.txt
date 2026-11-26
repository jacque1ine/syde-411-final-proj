TORONTO REAL-DATA TEACHING MODEL

This is a real-data adaptation, not the original Yao et al. experiment and not
an operational station recommendation. No NSGA-II has been run here.

REFERENCE
Yao, Zhang & Murray (2019), Location optimization of urban fire stations:
Access and service coverage. Data section, manuscript p. 10.
https://eprints.gla.ac.uk/172206/1/172206.pdf
The retained definition is w = fires / population * 1000, with 1 km² cells,
cell-centre candidate sites and 4 km straight-line coverage. The original
used 658 Nanjing cells, actual census population and 2002-2013 fire records.

OFFICIAL SOURCES (downloaded for this model)
https://open.toronto.ca/dataset/fire-incidents/
https://open.toronto.ca/dataset/fire-station-locations/
https://open.toronto.ca/dataset/neighbourhood-profiles/
https://open.toronto.ca/dataset/neighbourhoods/
Licence: Open Government Licence - Toronto. Source metadata and downloads
are preserved in raw/; summary.json records their SHA-256 fingerprints.
fetch.py retrieves the files from the official CKAN API.

STUDY AREA
225 cells, each 1000 m x 1000 m, in NAD83 / UTM zone 17N (EPSG:26917).
Southwest corner is (621500, 4837500) metres; rows increase northward and
columns eastward. Approximate study centre: longitude -79.40, latitude 43.75.
210 cells have nonzero estimated Toronto population and over 0.001 km² of
Toronto overlap. 14 cells lie outside Toronto; one additional cell with only
17 m² overlap (0.12 estimated residents) is excluded as a boundary sliver.
All 15 excluded cells have zero recorded fires. Border cells retain their
full 1 km² geometry but population includes only the Toronto overlap.

POPULATION ESTIMATE (NOT OBSERVED GRID POPULATION)
The 2021 158-neighbourhood profile's exact input variable is:
'Total - Age groups of the population - 25% sample data'.
These are census sample-based population estimates, not raw sample counts;
they must NOT be multiplied by four. They are a proxy for resident population,
not a full 100% census population grid. Uniform population density is assumed
within each neighbourhood polygon. Each cell receives the neighbourhood
estimate multiplied by its intersecting area / neighbourhood area, summed
over intersecting neighbourhoods. Fractional estimates are preserved.
This allocation spreads residents into parks and nonresidential land and can
bias per-capita rates. Dissemination-block census totals or a validated
population raster would improve fidelity to the paper.

FIRES
2018-2023 inclusive, six complete years, frozen from the published dataset
that extends through 2024. Includes Final_Incident_Type 01 (Fire) and 03
(no-loss outdoor fire); excludes type 02 (Explosion). Incident_Number is
unique in the selected records. Missing and invalid coordinates are excluded
and counted in summary.json. Coordinates are rounded by the publisher to
nearest major/minor intersections, and some records are withheld under
MFIPPA. Counts are thus published recorded fires, not all true fires.
Counts are assigned to grid cells with half-open bounds [west,east) and
[south,north). No simulated incidents have been substituted for real records.

RISK
w_i = 1000 * recorded_fires_2018_2023 / estimated_population_2021.
Units: recorded fires per 1,000 estimated residents over SIX years, not per
year or a probability. Population is held fixed at its 2021 estimate.
Zero-population cells have undefined (NaN) risk and are excluded explicitly.
The excluded boundary sliver has a defined zero rate but is not demand.
Small population denominators and areal interpolation can inflate rates.
Period totals cannot be compared directly with the prior synthetic 12-year
rates or the paper's 12-year rates without accounting for period length.

STATIONS
Published locations are filtered to TYPE_DESC == 'Fire Station'; training
facilities are excluded. 28 are inside the study square. The separate
baseline distance column uses all 84 published Toronto station locations,
including stations outside the square, to avoid an artificial edge penalty.
The station file is the downloaded publication snapshot, not a reconstruction
of station locations in each year from 2018-2023.

ILLUSTRATIVE REDESIGN
Candidate sites: active cell centres. Start with no existing stations, choose
the site covering the most still-uncovered centres within 4 km, repeat until
all active centres are covered. Then perform the best feasible one-for-one
station swaps to reduce sum(w_i * nearest_distance_i), stopping at a local
optimum or 40 swaps. Deterministic tie-breaking uses row-major cell order.
10 selected stations achieve 3.606 km maximum centre-to-nearest distance.
This is a feasible heuristic demonstration, NOT a proven minimum, a Pareto
frontier, NSGA-II, or the paper's exact solver. Existing stations are not
retained in this hypothetical redesign. Cell-centre coverage does not mean
every address within each cell is covered. Real station needs also depend
on roads, response time, apparatus, staffing and simultaneous demand; these
are not modeled. Do not interpret 10 versus 28 as evidence of excess stations.

REPRODUCE
Install numpy pandas openpyxl matplotlib requests shapely pyproj scipy.
Run python3 fetch.py, then python3 reproduce.py.
For the exact saved source snapshot, keep raw/ unchanged: future downloads
can differ. CSV exports retain full precision; displayed values are rounded.
