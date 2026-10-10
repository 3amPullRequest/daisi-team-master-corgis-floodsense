# Databricks AI Social Impact (DAISI) Challenge | Team Master Corgis
### *Track B2: FloodSense: Flash flood prediction & urban drainage intelligence*

# Problem Statement
PUB Flood Alerts, PUB’s official flood alert channel, provides flash-flood warnings that are reactive, giving too little lead time for those at risk to prepare.

Singapore recorded its wettest March on record in 2025, and its 7th-wettest year since 1980; annual rainfall has risen by about 92.5mm every 10 years since 1980. PUB drainage is comprehensive, but extreme rainfall is growing more common.

Predictive flood intelligence for when floods will happen at the zonal level would help residents prepare, but no available tool for this exists. 
The closest is Google’s March 2026 Urban Flash Flood forecasts on Flood Hub: they look 24 hours ahead on a 20 km × 20 km grid,
updated daily, so all of Singapore falls into a few cells. That is too coarse and too slow for the localised, short flash floods that occur in Singapore.


* For the 6 flash floods posted on PUB Flood Alerts from 22–30 September 2026, PUB’s first alert (the 90% drain water-level warning) came on average 13.3 minutes (min. 5) before the flood: less than half the median 30-minute car commute (2025 General Household Survey), too late for a driver to decide before setting off.
* Shop owners, low-floor residents, town councils and agencies are just as short of time to act, across ~15 flash floods a year (108 in 2017–2023, Environmental Research Letters, 2024) and 23.3 ha that PUB still designates flood-prone (2025, down only from 27 ha in 2022; Flood Prone Areas, data.gov.sg).

# Target Solution
FloodSense aims to forecast, every 5 minutes, which of NEA’s 47 weather-forecast areas is likely to get a PUB flash-flood alert in the next 60 minutes, rating each Low, Medium or High.