# Developer notes
## Overview of Architecture

```mermaid
graph LR
data_ingestion["Data Ingestion"]
-->
data_engineering["Data Engineering"]
-->
machine_learning["Machine Learning"]
-->
analytics_and_bi["Analytics and BI"]
-->
governance["Governance (Stretch)"]
```

### Data Ingestion
---

#### Labels
| Response | Type | Live Dataset | Historical Equivalent | Time Range | Time Interval | Live Source | Historical Source  | Remarks |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Flash Floods| Boolean | PUB Flood Alerts (Live API)   | Same | May 22, 2025 to Present | Minutes | [PUB](https://data.gov.sg/datasets/d_f1404e08587ce555b9ea3f565e2eb9a3/view) | NA | [See Appendix](#flash-floods-live-api) |
|Flash Floods & Water Level Sensor alerts (inferred to trigger at 90% water level) | Boolean | PUB Flood Alerts (Telegram)   | Same   | February 25, 2022 to Present| 1 Minute for Training, 5 Minutes for Live | [Telegram PUB Flood Alerts](https://t.me/s/pubfloodalerts?before=1632)   | NA | [See Appendix](#flash-floods--water-level-sensor-alerts) |



#### Features
| Feature | Type | Live Dataset | Historical Equivalent | Time Range | Time Interval | Live Source | Historical Source | Remarks |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Rainfall | Numerical | NEA Rainfall across Singapore (2016 to Present) | NEA Historical Rainfall (2016 to 2024) | 2016 to Present | 5 minutes | [NEA](https://data.gov.sg/datasets/d_6580738cdd7db79374ed3152159fbd69/view) | [NEA](https://data.gov.sg/collections/2279/view) | [See Appendix](#rainfall) |
| 2-Hour Weather Forecast | Categorical ([See Appendix](#2-hour-weather-forecast-possible-categories)) | NEA 2-Hour Weather Forecast (2016 to Present) | NEA Historical 2-Hour Weather Forecast (2016 to 2024) | 2016 to Present | 30 minutes | [NEA](https://data.gov.sg/datasets/d_3f9e064e25005b0e42969944ccaf2e7a/view) | [NEA](https://data.gov.sg/collections/2179/view) | [See Appendix](#2-hour-weather-forecast-time-interval) |
| Air Temperature | Numerical | NEA Air Temperature across Singapore (May 2016 to Present) | NEA Historical Air Temperature across Singapore (2016 to 2024) | May 2016 to Present | 1 minute | [NEA](https://data.gov.sg/datasets/d_66b77726bbae1b33f218db60ff5861f0/view) | [NEA](https://data.gov.sg/collections/2246/view) | [See Appendix](#air-temperature) |
| Relative Humidity | Numerical | NEA Relative Humidity across Singapore (2016 to Present) | NEA Historical Relative Humidity across Singapore (2016 to 2024) | May 2016 to Present | 1 minute | [NEA](https://data.gov.sg/datasets/d_2d3b0c4da128a9a59efca806441e1429/view) | [NEA](https://data.gov.sg/collections/2278/view) | [See Appendix](#relative-humidity) |
| Wind Direction | Numerical (circular) | NEA Wind Direction across Singapore (2016 to Present) | NEA Historical Wind Direction across Singapore (2016 to 2024) | Nov 2016 to Present | 1 minute | [NEA](https://data.gov.sg/datasets/d_534cf203023b51f51f879145ccc56ff9/view) | [NEA](https://data.gov.sg/collections/2281/view) | [See Appendix](#wind-direction) |
| Speed | Numerical | NEA Wind Speed across Singapore (2016 to Present) | NEA Historical Wind Speed across Singapore (2016 to 2024) | Dec 2016 to Present | 1 minute | [NEA](https://data.gov.sg/datasets/d_7677738484067741bf3b56ab5d69c7e9/view) | [NEA](https://data.gov.sg/collections/2280/view) | [See Appendix](#wind-speed) |
| Tide levels | Numerical | NEA Tide Timings | UHSLC Tanjong Pagar (imputed via utide) | 2016 to Present (UHSLC span unconfirmed) | Live: high/low times (about 4 a day); Historical: hourly | [NEA](https://www.nea.gov.sg/weather/tide-timings) | [UHSLC](https://uhslc.soest.hawaii.edu/data/download/fd#uh699) | [See Appendix](#tide-levels) |
| URA Master Plan | Categorical (land-use zoning polygons) | URA Master Plan 2025 Land Use Layer | URA Master Plan 2019 Land Use layer | Nov 2019 to Present | Static (new Master Plan roughly every 5 years) | [URA](https://data.gov.sg/datasets/d_a8c3546b26712e35021f3a681d0353ae/view) | [URA](https://data.gov.sg/datasets/d_90d86daa5bfaa371668b84fa5f01424f/view) | [See Appendix](#ura-master-plan) |
| Drainang Catchment Maps   | Categorical (catchment polygons) | NA (static) | PUB Drainage Catchment Map + Drainage Boundary (Northern & Southern Coast) Map | Static | Static | NA | [PUB](https://www.pub.gov.sg/Professionals/Requirements/Qualified-Persons/Catchment-Maps) | [See Appendix](#drainage-catchment-maps) |
| Elevation   | Numerical (raster, m) | NA (static) | SLA elevation (see Remarks); fallback: Copernicus GLO-30 DEM | Static (GLO-30 built from 2011–2015 TanDEM-X data) | Static | NA | [Copernicus](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) | [See Appendix](#elevation) |

#### Validation

| Variable | Type | Live Dataset | Historical Equivalent | Time Range | Time Interval | Live Source | Historical Source | Remarks |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| List of Flood Prone Areas | Categorical (36 named locations); Numerical (total ha) | NA | PUB List of Flood Prone Areas (Apr 2025); PUB Flood Prone Areas (annual, 2022–2025) | 2022 to 2025 | Annual | NA | [PUB list](https://pub.gov.sg/-/media/PUB/PDF/Flood-Management/List-of-Flood-Prone-Areas-as-at-Apr-2025.pdf), [PUB hectares](https://data.gov.sg/datasets/d_c4aed98f1533eb3a66f65dbb1a30da46/view) | [See Appendix](#list-of-flood-prone-areas) |
| PUB Water Level Sensors Locations | Geospatial (points) | NA | PUB Water Level Sensors | Snapshot (last updated 13 Apr 2026) | Refreshed half-yearly | NA | [PUB](https://data.gov.sg/collections/1541/view) | [See Appendix](#pub-water-level-sensors-locations) |



# Appendix

## Flash Floods (Live API)

PUB describes this as a real-time feed, so land every 5-min poll in bronze rather than relying on the API's lookback. Its description says flood warning notices are not included, so check which alert types (flash flood vs. 90% water level) it actually returns before using it for the Medium tier.

## Flash Floods & Water Level Sensor Alerts
There is much more relevant data, and as far as I can find this is the only source of water level sensor alerts (so we can train the model on this), but this data is really dirty. Alerts are inconsistent, and it isn't clear if the water level sensors has existed yet. We can compare against OpenMap, 2019 Master Plan, News Sources, PUB announcements, pub water level sensors dataset, or others, just to confirm in which location does the water level sensors exist yet. In the worse case, we can fit the model for those locations anyway on the assumption that they do have water level sensors, and use Bühlmann-Straub credibility to get a better estimate (I think. I could be mistaken on the application of credibility theory here).

## Rainfall

5-minute totals in mm per rain gauge (60 gauges on 1 Mar 2025; the station list changes over time). Historical collection ends Dec 2024 and is not quality-controlled. Backfill Jan 2025 onwards from the live API with `?date=YYYY-MM-DD`, following `paginationToken` (25 readings per page). Use a registered API key (rate limits since 31 Dec 2025).

## 2-Hour Weather Forecast Possible Categories
- Fair
- Fair (Day)
- Fair (Night)
- Fair and Warm
- Partly Cloudy
- Partly Cloudy (Day)
- Partly Cloudy (Night)
- Cloudy
- Hazy
- Slightly Hazy
- Windy
- Mist
- Fog
- Light Rain
- Moderate Rain
- Heavy Rain
- Passing Showers
- Light Showers
- Showers
- Heavy Showers
- Thundery Showers
- Heavy Thundery Showers
- Heavy Thundery Showers with Gusty Winds
## 2-Hour Weather Forecast Time Interval
Ad-hoc updates to the Weather Forecast may occur, making the time interval effectively an approximation. Hence, we should lower the time interval, as the chahnce of a forecast change is fairly low.

## Air Temperature

°C at 15 stations (1 Mar 2025), so it needs interpolating to the 47 areas. Historical collection ends Dec 2024; backfill 2025 onwards from the live API with `?date=` (1-minute data is ~58 pages per day).

## Relative Humidity

% at the same 15 stations as air temperature. Historical collection ends Dec 2024; backfill 2025 onwards from the live API with `?date=`.

## Wind Direction

Degrees, 10-minute mean reported every minute, 15 stations. Encode as sin/cos (359° and 1° are neighbours), or combine with speed into u/v components for the "upwind rainfall" features. Historical collection ends Dec 2024; backfill 2025 onwards from the live API with `?date=`.

## Wind Speed

Knots, 10-minute mean reported every minute, 15 stations. Historical collection ends Dec 2024; backfill 2025 onwards from the live API with `?date=`.

## Tide Levels

NEA republishes MPA's predicted high/low water (m above Chart Datum) for the current and next month only, as an HTML table with no API, so scrape it monthly. UHSLC station 699 is Tanjong Pagar (fast-delivery hourly sea level in mm, GMT timestamps): fit utide on it, then check its predicted highs/lows against NEA's table to fix the datum offset. UHSLC blocks automated access, so the exact span of the 699 series still needs checking by hand.

## URA Master Plan

GeoJSON polygons, zoning class in `LU_DESC`. MP2019 was approved 18 Nov 2019 and MP2025 gazetted 1 Dec 2025, so join MP2019 to rows before 1 Dec 2025 and MP2025 after (point-in-time). Zoning is planned use, not current land cover. Interim MP2019 amendments are a separate data.gov.sg collection if needed.

## Drainage Catchment Maps

11 drainage catchments (Jurong, Kranji, Pandan, Woodlands, Kallang, Bukit Timah, Singapore River, Stamford Marina, Geylang, Punggol, Changi). The catchment map is PDF only (image file names suggest a 2013 vintage), so polygons must be digitised and georeferenced by hand; the coastal boundary line also comes as DGN/DWG. 11 catchments vs. 47 areas, so best used for upstream-catchment grouping rather than as a fine-grained feature.

## Elevation

Couldn't find an openly downloadable SLA elevation model (data.gov.sg only has SLA vertical-control benchmark points); if the slide's SLA link is a specific product, swap it in here. GLO-30 is free with registration, 30 m, a surface model (includes buildings and trees), heights relative to EGM2008, and requires the Copernicus attribution notice. Use area-level summaries (min, mean, share of low-lying cells) rather than point values.

## List of Flood Prone Areas

The list (as at 2 Apr 2025) is 36 road/estate names with no geometry, so geocode with OneMap and assign each to an area. PUB defines these as low-lying areas with a history of flooding. The data.gov.sg file is only the yearly total (27 ha in 2022 down to 23.3 ha in 2025); an earlier version of the same dataset carried 2010–2021.

## PUB Water Level Sensors Locations

Seems to be down as of 11 October, 2026, 3:24pm. Iirc it was up when designing the model for Stage 1. Update: the data.gov.sg download API (`poll-download` for `d_31333fa5cf0834f012d840365b336610`) returned a working .xlsx link at 6:00pm SGT on 11 Oct 2026, so it may only be the web page that's down. It's a current snapshot, so it can't show when each sensor was installed (see [Flash Floods & Water Level Sensor Alerts](#flash-floods--water-level-sensor-alerts)).