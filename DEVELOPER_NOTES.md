# Developer notes

# BIG FINDING: PUB WIPES ITS TELEGRAM CHANNEL FROM TIME TO TIME (AT LEAST 4 TIMES SINCE 2022). ON THE CHANNEL, EVERYTHING BEFORE 19 MARCH 2025 IS GONE EXCEPT 16 POSTS.
# THE INTERNET ARCHIVE HOLDS MOST OF IT: `backfill.py telegram-wayback` RECOVERS 832 OF THE 1,092 MISSING POSTS (NOVEMBER 2021 TO MARCH 2025).
# ON 14 NOVEMBER 2023 THE CHANNEL MOVED FROM t.me/sgflood TO t.me/pubfloodalerts, TITLED "Gov.sg-PUB Flood Alerts" UNTIL 3 APRIL 2024. POST IDS DID NOT CHANGE.
# LOST FOR GOOD: FEBRUARY TO ~AUGUST 2024 (IDS 766-894) AND 5-19 OCTOBER 2024 (IDS 927-955) WERE WIPED BEFORE THE ARCHIVE CAPTURED THEM.

## Developer Checklist

Round 2 is a mentored build sprint from 12 to 26 Oct 2026. Demo Day is on 27 Oct, 4–7pm SGT, and picks the top 3; the finale is on 4 Nov ([DAISI guide](https://daisi.online/guide)). Demo Day is scored on social impact (30%), technical execution on Databricks (30%), solution and user experience (20%), data rigour (10%) and presentation (10%). B2's "winning demo" list asks for:

- a Low/Medium/High classifier by zone;
- a live risk dashboard that updates as rain streams in;
- a three-year trend of flood-prone areas;
- both batch data (training) and streaming data (inference).

Tags follow the Round 1 plan: **W1** is due Sun 18 Oct, **W2** is due Sun 25 Oct, and **Stretch** waits until both are done. Ticked boxes were checked against the repo and the dev deploy on 13 Oct, the silver ones on 14 Oct (`uv run pytest`: 115 passed), and the gold and ML ones on 15 Oct (160 passed), as were the silver tables for the other data.gov.sg sources (164 passed). An unticked box is either not done or still needs checking in the workspace.

### Data Ingestion

The bronze fetchers are written, tested and deployed to dev (see [Ingestion (bronze)](#ingestion-bronze)). What's left is running them to completion, plus the label work.

- [ ] Flash Floods
    - [ ] PUB Flood Alerts Telegram fully extracted for backfill **W1**
        - [x] `backfill.py telegram` pages back with `?before=` to the oldest visible post and lands one file under `pub_telegram_posts/archive/`.
        - [x] Archive landed in dev and visible in `bronze.pub_telegram_posts` (`source_file LIKE '%/archive/%'`): both scrapes, 10 Oct (522 of ids 1–1631) and 13 Oct (524 of ids 1–1633).
        - [ ] Record the missing ids at scrape time. The run summary's `posts_by_year` is a quick check.
        - [x] Counts match the 12 Oct label census (checked in `silver.telegram_posts`, 14 Oct). 2026: 120 warnings on 35 days and 8 flash floods on 6 days, exactly. Mar–Dec 2025: 137 warning posts on 44 days and 18 flash floods on 8 days (the census, 134 on 43 and 21 on 11, also counted older posts still on the channel).
        - [x] Posts PUB deleted, recovered from web archive captures with `backfill.py telegram-wayback` (13 Oct, locally into `.landing-dev`): 872 post ids from 93 captures in 12 minutes, including 832 of the 1,092 ids the channel no longer shows. By keyword, that adds about 435 "Risk of Flash Floods" warnings and 14 flash floods on 11 days (Nov 2021 to Mar 2025) to a training set of 21 floods.
            - [x] Landed in dev on 14 Oct: `bronze.pub_telegram_posts` holds 1,356 distinct ids of 1–1633 (wayback 872, archive 524, live 20).
            - [x] Label census with them (`silver.telegram_posts`, 14 Oct): before Mar 2025, 435 warning posts (555 locations) on 121 days and 16 flash-flood posts (17 locations) on 12 days.
            - [x] Decide whether training can start before Mar 2025: yes (15 Oct). Labels run from 27 Oct 2021, with the time posts were never recovered marked missing rather than quiet (`gold.label_ticks`), and the split is 80/20 of the recorded time (see [Gold](#gold)).
    - [ ] PUB Flood Alerts Telegram Live Ingestion fully functional **W1**
        - [x] `flood-live` lands posts newer than the cursor in `_state/telegram.json`, so it keeps posts that PUB later deletes.
        - [ ] Unpause `flood-live` in dev (the 10 Oct deploy left it paused). Then check that the cursor advances and that `pub_telegram_posts` reports `ok` in `bronze.ingest_heartbeat`.
        - [ ] Catch up after an outage. A tick reads only the newest page (about 20 posts), so after a gap, such as a day of exhausted quota, the posts in between are missed until the archive is re-run (if PUB hasn't deleted them by then). When the page doesn't reach back to the cursor, page back with `?before=` until it does.
        - [ ] Capture edits. Posts at or below the cursor are never landed again, so a later edit by PUB is missed. Keep text hashes of the newest page's posts in the cursor, and land a post again when its hash changes. Silver keeps the latest version.
    - [ ] PUB Flood Alerts Telegram Live Ingestion, Jev fallback for pattern matching failure, complete with warnings to the developer
        - [x] Categories: `warning` (Risk of Flash Floods, the 90% drain-level warning), `flash_flood`, `flood_subsided`, `heavy_rain`, `heavy_rain_eased` and `not_relevant`, plus `unmatched` for a post no pattern fits. **W1**
        - [x] Regex patterns for every category, one per format PUB has used (10 formats since Nov 2021), in `labels/alert_text.py`, which silver uses and a fetcher can import. Tested on 38 real posts in `tests/fixtures/pub_telegram_texts.jsonl`, including the drift found: 18 stamp spellings (`[14:13hours]`, `(Issued 11:00 hrs)`, `Issued 1510 hours`, `[Issued by NEA, 15: 30 hours]`, ...), a bullet left out (id 424), a doubled stamp (id 89), `[Risk of Flash Flood]` singular (id 1374), a window ending at 24:00 (id 323). **W1**
        - [x] Coverage report over the whole archive: `uv run python -m daisi_team_master_corgis_floodsense.labels.coverage .landing-dev/pub_telegram_posts`. Of 1,356 posts, all but 6 match a format; those 6 are announcements (scam advisories, the OBU announcement, a monsoon advisory, tips), labelled `not_relevant` in `data/static/telegram_manual_labels.csv`. **W1**
        - [x] Heavy Rain Warning and Heavy Rain Eased: detected by pattern (with their regions, window and NEA stamp in `silver.telegram_posts`) and kept out of `silver.area_alerts`. **W1**
        - [x] Jev, not Laya (the user's choice, 23 Oct): TypeSafe AI's Jev 1.14.0, which Databricks serves as `system.ai.typesafeai-jev-1-14-0` through Unity Gateway, so there is no checkpoint to load or cache. Jev answers questions with probabilities and writes no text, so it picks rather than reads: `labels/jev.py` cuts the post into candidate place phrases without knowing its format, and one request asks the category (a choice over the six) and, per candidate, whether it is a place (a yes/no). The stamp regex times each place. See [When no pattern reads a post](#when-no-pattern-reads-a-post-jev).
            - [x] Candidates checked over the archive without calling Jev (23 Oct): all 947 places the patterns read are among them (counting the 30 posts where `warning_bracket` reads "A; B" as one place), at 2–8 candidates a post, and timing them gives every one the pattern's stamp.
            - [x] In the live path: flood-live's `fetch` asks Jev once per post and text, lands every call in `bronze.telegram_jev_parses` (the record of each use of Jev), and silver uses a sure reading (category at least 0.6, place at least 0.5) where no pattern reads the post, as `method = 'jev'`, so the alert reaches `silver.area_alerts` (as a training label, `read_by = 'jev'`) and the app. Once a pattern exists, silver reclassifies the post by pattern on its next refresh; Jev's call stays in bronze.
            - [x] **Jev stays off (the user's decision, 23 Oct: Databricks charges for it).** Its endpoint isn't served here: Unity Gateway answers `404 ENDPOINT_NOT_FOUND` (free), and `databricks ai-gateway list-model-services --parent schemas/system.ai` lists no System One model; serving it needs the account-level preview **Externally-hosted TypeSafe AI Models**. No valid endpoint counts as a Jev error (`JevUnavailable`): it is landed with `status = 'error'`, the email says Jev failed and why, and the post's label windows go missing (`post not read`). Calls so far: 3, all free 404s on the made-up post (`purpose = 'test'`, 23 Oct 09:52, 10:34 and 11:13 SGT).
            - [ ] If Jev is ever turned on: validate the cut-offs on a handful of real posts the patterns already read (one paid request each), not the archive.
        - [x] Warn the developer on every fallback (`ingest/notify.py`). For now (the user's call, 23 Oct) the email says two things: the message the RegEx couldn't parse (id, t.me link, posting time, full text, and why: no headline matched, or a format matched with no location), and whether Jev failed (it couldn't be asked, it isn't sure enough of the kind of post, or it found no place) or what it read. Also a `fallback` summary in the heartbeat's Telegram result and the whole email as a `WARNING` in the job log.
            - [x] By SMTP, not a job notification: a Python task's failure email says only "Workload failed, see run output for details", so it can't carry the post. Gmail's SMTP is reachable from Free Edition serverless (probed 23 Oct, ports 465 and 587).
            - [ ] **The sending account** is in the scope (23 Oct), but Gmail refuses its login: the stored `parser-alert-smtp-user` isn't a full address. Store the address and its 16-letter app password again, then send the test alert (`src/ingestion/check_parser_alert.py`, [One-time setup](#one-time-setup)). Until it works, alerts are logged in full, retried for an hour of ticks, then dropped.
            - [x] Recipients: the `parser_alert_to` bundle variable (stkinzheng, frostlynx6, eugenehengruizhe), passed to `fetch` as `--alert-to`.
    - [ ] PUB flood-alerts API (flash floods only, with lat/lon circles) **W1**
        - [x] Live ticks, and `api-days` from 2025-09-01. Real records start around Oct 2025.
        - [ ] Day backfill finished. The 10 Oct laptop run reached 2026-01-10 and stopped (see the next block).
        - [x] Check which alert kinds and statuses it returns, and how a record repeats while an alert stays active, so silver counts each alert once. Up to 2026-01-10 it holds 68,541 two-minute records, of which 6 carry an alert (3 floods and their 3 "subsided" notices, on 4 and 12 Dec 2025), all headlined "Flash Flood Alert"; only the description tells a flood from its end. A record comes back on every tick until it scrolls off the newest page, so silver keeps one per issue time and description.
- [ ] Weather time series: rainfall, 2-hour forecast, wind direction and speed, lightning, air temperature, relative humidity, radar
    - [x] Fetchers: live ticks and day backfill for all nine API sources, and yearly CSVs for the six that have them.
    - [ ] Restart the laptop `api-days` backfill with an API key **W1**:
        - [x] Register a data.gov.sg key and store it in the `floodsense` secret scope (see [One-time setup](#one-time-setup)). A prod key since 15 Oct: the laptop run reads it from the scope, and in 9 hours at 60% of the prod limit it met no HTTP 429 (a key data.gov.sg doesn't recognise gets the no-key limit, 6 calls per 10 s).
        - [x] Laptop runs read the key from the secret scope too (through `--profile`), and `check_key.py` measures which tier a key really gets.
        - [ ] Since 11:56 on 15 Oct the laptop run has a prod key (`--key-tier prod`, log `~/floodsense-api-days.log`). By 21:10 it had landed every day through 14 Oct 2026 of the labels (`pub_flood_alerts`, 409 days from 2025-09-01), rainfall and the forecast (652 days each from 2025-01-01), checked on the volume. The 1-minute sources take about 33 s a day, so about 6 hours each: wind direction (at 30 May 2025 by then), wind speed, lightning, air temperature, humidity, then radar. The run executes from the project `.venv`: rebuilding it underneath killed an earlier run on 15 Oct (`certifi` lost its path), so while it runs, test with `.venv/bin/python -m pytest` and leave `uv sync` for later.
        - [ ] It resumes where it stopped. With a dev key, labels, rainfall and the forecast land in about 8 hours, and everything in about 3 days.
    - [x] Run the `hist_csv` task of `flood-backfill` (2021–2024): all six sources landed on 10 Oct, four years each (checked 15 Oct). Re-run it when data.gov.sg publishes 2025. **W1**
    - [ ] Check coverage per source and day with the progress query in [Checking it](#checking-it). Rainfall must cover every day from the training start, 2021-10-27 (the CSVs cover 2021–2024; the API backfill the rest). **W1**
    - [ ] Unpause `flood-daily`, so each night repairs gaps in the live feed. **W1**
    - [ ] Radar starts on 2026-07-01, so a model trained on 2025 can't use it: archive it only for now (**Stretch**). Lightning starts on 2025-02-01, so it covers the last 10½ months of training (Feb–Dec 2025) and all of the test; it reaches silver as `silver.lightning_strikes`.
- [ ] Tides
    - [x] NEA tide table, scraped daily (this month and next); UHSLC Tanjong Pagar hourly, 1984–2023.
    - [ ] Run the `uhslc` task, then check its row count and the `-32767` gaps. **W1** Landed locally on 20 Oct: 350,632 hours (1984-01-01 to 2023-12-31 15:00 GMT), 12,386 of them `-32767`, almost all before 2016.
- [ ] Static layers
    - [x] `fetch_static.py`: URA MP2019 and MP2025 land use, PUB water-level sensors, flood-prone area (ha), SLA vertical-control benchmarks, and (25 Oct) the national map's polygons and lines.
    - [x] Run the `static` task, and confirm each `static_*` table holds one `version`: landed on 10 Oct, one version each (checked 15 Oct). **W1**
    - [x] The national map (25 Oct): `static_national_map_polygon` (1,585 polygons: coastal outlines, hydrographic features, parks, runways, the CBD) and `static_national_map_line` (17,070 lines: contours every 20 m, expressways and slip roads, major roads, the international boundary). Landed by the `static` task with `--layers` (`bundle run backfill --only static --python-params ...,--layers,static_national_map_polygon,static_national_map_line`), because a plain run would also have landed MP2019's newer version (18 Oct 2026). The contours carry a vertex every 30 cm or so, up to about 900,000 in one line, more than one bronze VARIANT holds (16 MiB). So a line of more than 50,000 vertices lands as consecutive parts (`static.feature_parts`): 17,102 landed lines, up to 19 parts each, none unparsed.
    - [ ] PUB list of flood-prone areas (Apr 2025 PDF, 36 names): transcribe it to a CSV seed in `data/static/`. Silver geocodes it. **W1**
    - [ ] NEA's 47 forecast areas and their label points come with the 2-hour forecast (`area_metadata`), so silver builds them; there's nothing extra to ingest.
    - [ ] Elevation: the national map's contours (25 Oct) give heights only every 20 m (its 1:250,000 layer), enough to tell the hills from the lowland but not the metre or two that makes a road low-lying. Finer heights need a DEM: the Copernicus GLO-30 (30 m raster; registration and an attribution notice required), a one-off download from a laptop into a UC volume. The SLA benchmarks are out: none of the 314 points carries a height, only its number, status, a checksum and its update time, and every z is 0 (checked 15 Oct, and again 25 Oct). **W2**
    - [ ] PUB drainage catchment maps: PDF only, so the 11 catchments have to be digitised by hand. Only upstream-catchment rainfall needs them. **Stretch**
- [ ] **Stretch:** LTA DataMall, for B2's ridership-disruption stretch goal. It needs an account key; first check what it offers at the time resolution of a storm.

### Data Engineering

Silver goes in the `flood-etl` pipeline and gold in `flood-gold`, under `src/pipeline/transformations/{silver,gold}/`. Build them backwards from `ml/contract.py`. All timestamps are SGT.

Silver is built (14 Oct) up to the labels and the two core feature sources, and since 15 Oct holds every other data.gov.sg source too: the weather readings and stations, lightning, land use and the flood-prone area; see [Silver](#silver). What's left there is the area polygons and the tides. The area context moves to gold, which maps feature sources to areas (as `gold.area_rain_stations` does for the gauges).

- [ ] Silver: reference tables **W1**
    - [x] `silver.forecast_areas`: the 47 areas and their label points, from `area_metadata` and the yearly CSVs. 47 rows.
    - [ ] Polygons for the app's map: Voronoi cells around the label points, clipped to the coastline (the union of the Master Plan land-use polygons works as one). Alerts don't need them: the nearest label point is the Voronoi cell.
    - [x] `silver.rain_stations`: the rain gauges, with location and first and last day seen: 104 ids (89 in the API's list on 10 Oct, 80 in the 2021–2024 CSVs). None has moved; some were renamed.
    - [x] `silver.weather_stations`: the automatic weather stations, with location, the variables each reports, and the first and last day seen. 22 on 15 Oct: the 19 of the 2021–2024 CSVs, plus S23 Tengah, S80 Sembawang and S199 from the API. **W2**
- [ ] Silver: readings
    - [x] `silver.rainfall_5min` (station, ts, mm), from live, backfill and yearly CSVs **W1**:
        - [x] explode `payload:body.data.readings` and cast;
        - [x] keep `status = 200` pages only (404 and empty pages are day markers);
        - [x] deduplicate on (station, ts), keeping the latest `fetched_at` (Auto CDC, so each update merges only new pages);
        - [x] expectations: a parsed reading, 0 ≤ mm ≤ 60 (the 2021–2024 maximum is 37.2 mm). All 27,329,166 readings passed.
        - [x] `published_at`: when NEA published each reading, from the CSVs. Gold needs it (see the reading-lag item below).
    - [x] `silver.weather_1min` (station, variable, ts, value): air temperature, humidity, wind speed and wind direction at NEA's 1-minute resolution, from the yearly CSVs, the API day backfill and live ticks (15 Oct). **W2**
        - [x] Not the 5-minute means first planned. The same reading lands many times, so it has to be deduplicated (Auto CDC) before anything is averaged; and an average of floats keeps a materialized view from refreshing incrementally, so a 5-minute view would rescan 112 million readings on every update. NEA's wind readings are already 10-minute means. The 5-minute values, and the wind as u/v, belong in gold, like the rain windows.
        - [x] Deduplicated on (station, variable, ts); within a CSV the later publication wins (88 revised readings: 42 temperatures, 46 humidities).
        - [x] Expectations: a parsed reading, and a value inside wide physical bounds (`utilities/silver_sources.py`; a test checks that the 2021–2024 extremes fall inside). All 112,345,530 readings passed (15 Oct).
        - [ ] The 2025–26 API days arrive with the laptop backfill (wind direction through May 2025 on 15 Oct); each `flood-etl` update merges what has landed.
    - [x] `silver.lightning_strikes`: one row per strike (time to the millisecond, place, `C` cloud to cloud or `G` cloud to ground), from the API; NEA's lightning starts on 1 Feb 2025. Deduplicated on (ts, lat, lon, type), since a record lists some strikes twice. Empty until the backfill reaches lightning (15 Oct). **W2**
    - [x] `silver.forecasts_2h` (area, issued at, published at, valid window, category), with NEA's code and an ordinal `rain_class` (0–5) for the categories. Joins use only forecasts with `updated_at` before the tick. 4,140,371 rows, every category known.
        - [x] Forecasts NEA issued from 1 Jan to about 7 May 2025 all carry 8 May 2025 (03:51–04:35 UTC) as `update_timestamp`: NEA re-published them in bulk, 100–160 days after issue (2021–2024: 3–9 minutes). Gold uses a forecast only from its `updated_at`, so those four months, the wettest March on record included, are blind for want of forecasts. That, not the backfill, is why most of Jan–Jun 2025 was blind on 15 Oct. Done 21 Oct, in gold (`features.published_s`): a forecast published in that batch (8 May 2025, 11:00–13:00 SGT, `contract.FORECASTS_REPUBLISHED_SGT`) counts from `issued_at` plus 10 minutes, which covers 99% of the API's other forecasts (p99 9.6 min, 2025–26). Every issue from 1 Jan 00:00 to 8 May 07:30 is in the batch, one version each, so no amendment is lost; Jan–Apr 2025 then have a forecast at 98.6–100% of ticks. NEA's other late publications (a few issues a month, 2–12 hours late, and 100 issues of 8–9 Jun 2025) were real outages and keep their times. No flag column: the imputed rows are exactly the batch's rows of `silver.forecasts_2h`. **W2**
    - [x] Tides (20 Oct; see [Tides](#tides)): `silver.tide_gauge_hourly` and `silver.tide_extremes` clean the two sources, and gold models and imputes:
        - [x] `gold.tide_model`: utide fitted on UHSLC 2005–2023, constituents kept.
        - [x] `gold.tide_levels`: a tide at every 5-minute tick, 2021 through next year, imputed from the prediction wherever the gauge is silent.
        - [x] `gold.tide_datum`: the datum offset against NEA's high and low waters (−0.005 m), with expectations at 0.15 m and 30 min (worst seen: 0.11 m and 19 min).
- [ ] Silver: labels **W1**
    - [x] `silver.telegram_posts`: one row per `message_id`, archive, wayback and live merged, latest version kept. Columns: category, method (`pattern` or `manual`; `laya` later), pattern, problem, hint, and the locations with their stamps.
        - [x] Stamp to datetime: the posting date, minus a day when the stamp is more than 10 minutes after the posting (posts just after midnight, e.g. ids 1122 and 1290).
        - [x] A post that names several locations becomes one alert per location (855 from 734 alert posts).
        - [x] `silver.telegram_review`: posts no pattern matches, or matched with a problem, each with the action that resolves it.
    - [x] `silver.api_flood_alerts`: issue time, category (read from the description), location, `alert_ts` from its stamp, circle centre and radius.
    - [ ] `silver.location_gazetteer` (normalised location text → lat/lon → area, method):
        - [x] Hand fixes first (`data/static/location_gazetteer_manual.csv`), then PUB's own circle from the API, then the water-level sensor of the same name (SVY21 converted to WGS84; a name match of at least 0.8, since lower scores land on the wrong stretch of long roads), then OneMap.
        - [x] OneMap search: `backfill.py onemap`, on a laptop. Without a token OneMap answers about one search every 10 s: 317 locations took about 2 hours over two runs (15 Oct). An answer counts only if its road, name or building is the place searched (OneMap's top answer for "Kings Rd" is a building on Joo Chiat Road). A re-run asks only what hasn't been asked.
        - [x] Misses go to `silver.alerts_unplaced`, with the best-named sensor as a hint (the Round 1 risk plan).
        - [x] Report the share geocoded by each method (query in [Checking it](#checking-it-1)); it's a data-rigour point for the judges. 15 Oct, after OneMap: of 858 alerts, OneMap placed 511 (60%), PUB's sensor names 290 (34%), PUB's API circles 7 (1%); 50 (6%, all warnings, at 29 locations) wait in `silver.alerts_unplaced`.
        - [ ] Hand-place the 29 unplaced locations in `data/static/location_gazetteer_manual.csv`: they blank their storm hours for every area (see [Gold](#gold)).
        - [ ] Hand-check the gazetteer, starting with the rows where OneMap disagrees with the chosen point (`onemap_km`).
    - [x] `silver.area_alerts`, exactly as `contract.AREA_ALERTS_COLUMNS` defines it, with the contract's descriptions as column comments:
        - [x] An API flood that is a Telegram post's alert (same kind, stamped within 15 minutes, named alike) becomes one row, with the Telegram stamp as `alert_ts`.
        - [x] Expectations: area not null (otherwise `silver.alerts_unplaced`), and kind is `warning` or `flash_flood`.
        - [x] Subsided and heavy-rain posts stay out.
    - [ ] Geocode the 36 flood-prone names with the same gazetteer (after transcribing them, see Static layers).
- [ ] Area context **W2**
    - [x] `silver.land_use`: both Master Plans, one row per polygon (226,606), with URA's land use, its area, the polygon as GeoJSON, and the days each plan is in force (15 Oct).
    - [x] `silver.flood_prone_area`: PUB's flood-prone hectares by year, 2022–2025 (15 Oct).
    - [x] `silver.map_polygons` and `silver.map_lines` (25 Oct): the national map, one row per polygon, line or landed part of a line, with its layer, name, SLA's length or area, the contour's height and the GeoJSON.
    - [x] `gold.area_terrain` (25 Oct): the ground of each area, over its land: how far the hills (contours of 20 m or more), the sea and the nearest canal or river are, how high the hills within 5 km rise, and the shares of land crossed by trunk roads or in parks. A model input only where a spec adds it (`add=map_line`, `add=map_polygon`); see [The ground](#the-ground-from-the-national-map).
    - [ ] `gold.area_static` (planned as silver, but it maps feature sources to areas, which is gold's job), one row per area and plan:
        - [ ] land-use shares (MP2019 before 1 Dec 2025, MP2025 after): `h3_polyfillash3` on `silver.land_use.geometry` at resolution 11 (about 0.002 km² a cell), each cell to the area of the nearest label point, then shares of cells. On serverless on 15 Oct a 0.01° square filled to 506 cells;
        - [ ] elevation summary (min, mean, share of low-lying cells), from GLO-30 once it lands;
        - [ ] number of water-level sensors;
        - [ ] number of flood-prone locations.
- [ ] Gold (`flood-gold`, built 15 Oct; see [Gold](#gold))
    - [x] `gold.area_features`: area × 5-minute tick, point-in-time. 18.0 million rows (2021–2024, and a live tick), every expectation passed.
        - [x] Contract columns: `rain_{15,30,60}min_mm`, the largest total among gauges within 5 km (`gold.area_rain_stations`: 493 pairs, every area has one), plus `rain_gauges`. The rule baseline needs only these. **W1**
        - [x] The reading lag, measured on the CSVs (`silver.rainfall_5min.published_at - ts`): NEA published every 2021–2024 reading more than 5 minutes after its timestamp, and 4–20% of rain readings more than 10 minutes after. So `READING_LAG_MINUTES` is now 10, and a reading also waits for its `published_at` where the CSVs give one. Checked against a hand recomputation from silver on five area-ticks: equal. **W1**
        - [ ] Measure the lag again from live ticks once `flood-live` runs (2025 on has no publish time). **W1**
        - [x] The 2-hour forecast: `forecast_rain_class` (0–5) and `forecast_age_min`, from the newest forecast published by the tick while it is valid (`gold.area_forecasts`). Checked against a hand recomputation from silver on 1,827 random area-ticks: equal. A third of the training ticks an alert followed had no rain within 5 km yet, and the forecast is what tells them from a quiet afternoon (see [Gold](#missing-features-versus-dry-ones)). **W1**
    - [x] `gold.feature_ticks`: per tick, whether the features are known. Missing when the gauge network didn't report at every reading timestamp the rain windows read, or no forecast was valid. An outage leaves no `area_features` row, which must not read as a dry tick. **W1**
    - [x] Labels: `gold.label_coverage` (id gaps as intervals), `gold.label_ticks` (per tick: recorded or missing, why, and the split) and `gold.area_labels` (per area and tick: `alert`, `no_alert` or `missing`, and 1/0/null labels for alerts and floods). **W1**
    - [x] The 80/20 temporal split on recorded label time: train 27 Oct 2021 – 18 Dec 2025, a day's embargo, test 20 Dec 2025 – 12 Oct 2026 (`contract.TEST_START`, computed by `split.temporal_split`). **W1**
    - [x] `gold.model_rows`: the only rows models learn from or are tested on, where both the label window and the feature window are recorded. Expectations fail the update if a missing interval gets in. **W1**
        - [ ] Columns for the gated model **W2** (16 Oct: the optional features, see [Gold](#rain-around-the-area-weather-and-lightning)):
            - [ ] antecedent rain (3 h, 24 h) and rain trend;
            - [x] rain 5–20 km away (`rain_nearby_30min_mm`), and the same upwind of the island-wide wind (`rain_upwind_30min_mm`). Of the 661 dry training rows an alert followed (no rain within 5 km in the hour), 76% had rain 5–20 km away, against 15% of quiet dry rows. Upwind is undefined when the mean wind is under 2 knots, which in Nov–Dec 2024 was 42% of the hour before alerts (storm winds point every way), so expect little from it;
            - [x] forecast category and its age (15 Oct);
            - [x] tide height and minutes to the next high tide (20 Oct; the predicted tide, see [Tides](#tides));
            - [x] temperature, humidity and wind, island-wide (`air_temp_c`, `rel_humidity_pct`, `wind_u_kn`, `wind_v_kn`; 19 stations can't give 47 areas their own). Known for 98–99% of 2021–2024; 2025–26 fills in as the weather backfill lands;
            - [x] lightning within 10 and 30 km in the last 30 minutes (`lightning_10km_30min`, `lightning_30km_30min`), from Feb 2025; empty until the lightning backfill lands. On 13 Apr 2025 Sembawang had 500 strikes within 10 km at 13:10 with 0.2 mm of rain; PUB's flood alert came at 13:50. In gold, but held back from the models (`contract.HELD_BACK_FEATURES`, 16 Oct): to feed it to them, drop it from there and give it its monotone constraint and reason in `gated.py`;
            - [x] the area context as `gold.area_land_use` (land-use shares per area and plan, by H3 cells). Not a model input: a share is the same at every tick of an area, and the models already read the area;
            - [ ] Per-area history, such as the past alert rate, is computed on the training window only.
        - [ ] Live/backfill parity: rebuild a live day from backfill files and compare. Replay is only honest if the two match. **W2**
    - [ ] `gold.flood_prone_trend` (year and hectares: 27 ha in 2022 down to 23.3 ha in 2025) and `gold.flood_prone_locations` (the 36 geocoded names). **W1**
        - [x] `gold.flood_prone_trend`: year, hectares, and the change from the year before (16 Oct).
        - [ ] `gold.flood_prone_locations`, once the 36 names are transcribed.
    - [x] `gold.pub_lead_times`: minutes from PUB's first alert to each flash flood in the archive, to re-measure the 13.3-minute figure as Round 1 promised. Built on 25 Oct as `ml.scoreboard_floods.pub_warning_min` instead (`flood-scoreboard`, from `scoreboard.alert_events`), beside FloodSense's lead before the same floods. **W1**
- [ ] Pipeline health
    - [ ] A `flood-etl` update has to finish well inside the 5-minute tick. Time it after adding each table, and move heavy rebuilds to `flood-daily` if it doesn't fit. **W1** The first silver build, including the one-off load of 27 million CSV readings, took 4.5 minutes on 14 Oct. A later update with no new readings took about 5 minutes: 2 waiting for and starting serverless compute, and 2.5 running, mostly the label tables (the gazetteer's sensor matching about 30 s, each Python-UDF table a few seconds to start its workers). That leaves no room in a 5-minute tick; options are a separate labels pipeline that `flood-live` refreshes only when new posts land, or a 10-minute cadence. Decide once gold is in. On 15 Oct the update that first built the weather tables (110 million CSV readings) and streamed the whole 2025–26 rain and forecast backfill ran 10.5 minutes (12 with start-up). What the 13 new flows add to a routine update isn't measured yet.
    - [x] Tests: stamp, location and alert parsers against fixtures, plus contract tests that silver's and gold's columns match `contract.py` (`tests/test_silver_contract.py`, `tests/test_gold_contract.py`). **W1**
    - [x] Gold in its own pipeline, so the 5-minute `flood-etl` updates don't rebuild years of features: the first `flood-gold` build took about 1.5 minutes of running (3.5 with compute start-up). `flood-daily` refreshes it after the repair. With the optional features (16 Oct) a full rebuild runs 3.6 minutes (4.4 with start-up). Live scoring computes the newest ticks' features from silver with gold's own definitions (`features.py`, the weather and the rain around the area included since the 21 Oct merge; see [Staying live](#staying-live)).
    - [ ] Feed expectation pass rates from the pipeline event log to the dashboard's health page. **W2**

### Machine Learning

`src/daisi_team_master_corgis_floodsense/ml/` holds the contract, scoreboard, rule baseline, time split, the gated LightGBM pair (`gated.py`) and the training job (`train.py`). Every model learns from `gold.model_rows`, is judged by `scoreboard.score` against `silver.area_alerts`, and writes `ml.predictions`. See [Machine learning](#machine-learning-1).

- [x] `contract.py`: 47 areas, 5-minute ticks, 60-minute horizon, a 10-minute reading lag, labels from 27 Oct 2021, the 80/20 split (test from 20 Dec 2025 to 13 Oct 2026), the gold label and feature-coverage tables, and `model_rows`.
- [x] `scoreboard.py`: events, episodes, lead time, false alarms per area-month, High precision, and 95% CIs (Wilson, and a bootstrap over days). `missing=` (the missing ticks of `gold.label_ticks`) and `blind=` (those of `gold.feature_ticks`) leave out what can't be judged, and count it in `*_unjudged`.
- [x] `baseline.py`: fits the rule's window and its Medium and High thresholds on the training window, honouring `missing=` and `blind=`.
    - [x] Smoke test on gold, 15 Oct, after geocoding (train split where gold had rain: 27 Oct 2021 – 22 May 2025, as far as the backfill had reached). Medium from `rain_15min_mm` ≥ 17.5, never High: 10.8% of 240 judged alert events flagged before PUB (95% CI 7.5–15.4%), 1.8 min mean lead, 0.83 false alarms per area-month. 160 events were unjudged: 114 next to id gaps (PUB deletes posts mostly during storms), the rest next to the 50 unplaced warnings. Not the real fit yet: that waits for the 2025 rain.
- [x] `split.py`: `temporal_split` (the 80/20 cut on recorded ticks), 5 blocked day-folds in the development period, a 1-day embargo, and the test kept for a single run.
- [ ] Setup **W1**
    - [x] `lightgbm` and `scikit-learn` in `pyproject.toml`; the training job adds `lightgbm` to serverless environment 6, which has scikit-learn and `mlflow-skinny`. `utide` is in `pyproject.toml` and in `flood-gold`'s environment (20 Oct). The spatial work needs neither `shapely` nor `pyproj`: Databricks' H3 and ST functions run on serverless (`h3_polyfillash3`, `st_geomfromgeojson`; checked 15 Oct through Databricks Connect), and SVY21 is converted in `labels/locations.py`.
    - [x] `resources/train.job.yml` (`flood-train`) and `resources/ml.yml` (the MLflow experiment and the UC registered model), deployed to dev on 15 Oct.
    - [x] Every run logs `split.summary(...)`, and the coverage of each split.
- [ ] Rule baseline end to end **W1**
    - [x] `flood-train` fits it on the development period and logs its parameters, scoreboard metrics and threshold curve to MLflow, next to the pair.
    - [ ] Write its predictions to `ml.predictions`. `flood-train` writes them for the test split when it scores it (`score_test=true`); live predictions come with the `score` task.
    - [ ] Add a `score` task to `flood-live`, after `refresh_pipeline`, that scores the newest tick for all 47 areas. The app then runs on the rule in Week 1. Written and tested locally on 21 Oct (`ml/live.py`; see [Staying live](#staying-live)); not yet deployed.
- [ ] Gated LightGBM pair **W2** (`ml/gated.py`, trained by `flood-train`; first trained 15 Oct on the training time gold could use then, Nov 2021 – Dec 2024)
    - [x] Gate: a LightGBM, not a rain threshold. A third of the training ticks an alert followed had no rain within 5 km yet, so any rain threshold drops the early ticks that give lead time; the gate reads the forecast and the hour too. It trains on every recorded tick (quiet ones subsampled to 5% and weighted back up), and its out-of-fold scores set its threshold: the highest that passes 90% of alert ticks. 15 Oct: it passes 12% of area-ticks and leaves 90% of PUB events catchable. The job tries gate recalls 0.85, 0.9 and 0.95, and the scoreboard picks one.
    - [x] Rater: LightGBM on the ticks the out-of-fold gate passed.
        - [x] Label: `gold.area_labels.alert_label` (what `scoreboard.tick_labels` computes), any PUB alert within 60 minutes. The 90% warnings train the Medium tier.
        - [x] Cross-validated with `split.cv(...)` and early stopping on the mean out-of-fold loss. No hyperparameter search yet.
        - [x] Monotone constraints: the alert chance never falls as rain or the forecast class rises.
    - [x] Calibration: centered isotonic regression on the pooled out-of-fold scores (plain isotonic gave thousands of storm ticks one probability, so a cut-off couldn't fall between them). The reliability table and Brier score are logged.
    - [x] Tier cut-offs, from the same pooled predictions:
        - [x] Medium: the most PUB events flagged before PUB, within no more false alarms than the rule raises (at most 1 per area-month);
        - [x] High: where at least 1 in 3 High episodes are followed by a flash flood (the same logic as `baseline.fit`). 15 Oct: never, as one flood event could be judged; the 2025 floods arrive with the backfill.
        - [x] Every cut-off lies above the probability of the rows the gate screens out (16 Oct). Calibration floors the passed rows at that probability, so the lowest candidate could sit on it and warn every row, and the scoreboard can't fault that: one endless warning per area holds every alert, so it is never a false alarm. The new features made the synthetic test's gate sharp enough to pick it.
    - [x] Reasons: LightGBM's `pred_contrib=True`, the two inputs that raised the score most, in plain English ("19 mm of rain within 5 km in the last 15 min; NEA forecasts thundery showers here"). Since 16 Oct also the rain 5–20 km away or upwind ("14 mm of rain upwind, 5-20 km away in the last 30 min"). Lightning and tides join when the models read them.
    - [x] Before touching 2026, beat the rule on both lead time and false alarms out of fold, as committed in Round 1. 15 Oct: 20.8% of 192 judged alert events flagged before PUB (95% CI 15.7–27.1%) against the rule's 9.4% (6.0–14.3%), 5.1 min mean lead against 0.9, at 0.87 false alarms per area-month against 0.91. Registered as version 2, `@champion`. Refit once the backfill lands: 2025 holds a third of the training time.
    - [x] Too conservative (22 Oct): the floods hardly reached training (16 of 390 flood rows), and the area read as a category crushed heavy rain in areas with few warnings. Recovered the data (unplaced alerts placed by hand, the Jan–May 2025 forecasts read from their issue), made the area a Bühlmann–Straub prior, and counted a flood flagged as 10 events in choosing. Out of fold: 30.1% of 322 events flagged before PUB and 4 of 13 floods, against 26.1% and 1 for the old pair on the same data. Not yet registered. See [Results so far](#results-so-far).
    - [ ] Decide whether single-post id gaps are posts PUB deleted (recorded) rather than lost (missing): about 10 more training floods (see [Results so far](#results-so-far)).
- [ ] Held-out test, run once **W2**
    - [ ] Score 2026 to date: `databricks bundle run train -t dev -p flood_dev --params score_test=true`, once the API backfill has landed rain and forecasts through 12 Oct 2026 and `flood-etl` and `flood-gold` have run. It refuses while under 95% of the test's recorded label time has features (0% on 15 Oct).
    - [x] Log the event-by-event table and the metrics with 95% CIs for the pair and the rule (PUB's own warning time is in `*_mean_pub_warning_min`), and write the test predictions to `ml.predictions`. Since 25 Oct `flood-scoreboard` publishes the scoreboard tables for the dashboard and Genie (see [For agencies](#for-agencies-the-scoreboard-and-genie)).
        - [x] The write, dry-run on serverless on 15 Oct (Databricks Connect, synthetic predictions into a scratch table, then dropped): one row per model and tick, `p_alert` NULL for the rule, SGT times unshifted, the contract's column comments, clustered by `(model, ts)`; a rerun keeps the other models' rows and replaces its own. `score_test` writes the predictions before it logs any test metric, so a failed write leaves the test unseen.
    - [ ] State the power caveat. The test has 8 floods, so even 8/8 gives a Wilson interval of 0.68–1.00, and the ">80% of floods flagged" target can't be confirmed on them alone. The alert-level metrics (120 warnings on 35 days) are the ones with usable CIs.
- [ ] Register and serve **W2**
    - [x] One pyfunc (`ml/serving.py`) holds the gate, rater, calibrator, cut-offs and reasons. Registered in UC as `<catalog>.<ml schema>.floodsense` (dev: `flood_dev.dev_stkinzheng_ml.dev_stkinzheng_floodsense`): every version `@challenger`, and `@champion` when it beats the rule out of fold.
    - [ ] Register the rule too, as the comparison.
    - [x] Switch the `flood-live` `score` task to `@champion`. It scores inside the job, with no serving endpoint (Free Edition limits those), and appends `model, area, ts, tier, p_alert, reason`. Cast the feature columns to double first: the model's signature refuses int64. Done in code on 21 Oct: `@champion` when it loads, else the newest trained rule; features cast to float64. Ran with v5 on 23 Oct.
    - [x] Verify before promoting (23 Oct): `flood-train --params promote=false`, then `flood-verify` (seven checks on the registered artifact, every flood side by side with the old pair and the rule), which moves `@champion`. v5 passed and is `@champion`. See [Verifying a registered pair](#verifying-a-registered-pair-before-it-goes-live).
    - [ ] Decide the Medium false-alarm budget: the rule's 0.87 per area-month catches 4 of 13 development floods before PUB; 2.7 would catch 9 (see [Results so far](#results-so-far)).
    - [ ] Live scorekeeping: a daily task scores every live prediction against the alerts that followed, the "keeps score in public" promise.
    - [ ] Storm replay: batch-score one flash-flood day from 22–30 Sep 2026 at every tick, and record how many minutes before PUB's first alert its area turned Medium.
- [x] Review of the model pipeline (23 Oct, [ARCHITECTURE.md](ARCHITECTURE.md)): the ordinal model (`ml/ordinal.py`: two rungs, P(alert) for Medium and P(flash flood) for High, Frank & Hall; Medium on the warning score p_alert + 9 p_flood), with or without the gate; penalized logistic regression as a learner (`ml/logistic.py`); the rule as every fit's safety override; the dew point and the prior rain (the `extended` feature set); PR-AUC and row recall logged for every model; `flood-train --params models=...` fits any mix. Out of fold nothing beat v5's design on the ranking; the hybrid (`ordinal:alert=gated:flood=logistic`) is better on the flood targets and is registered as `@challenger`.
- [x] Hybrid vs v5 as `@champion` (23 Oct, user's decision): the hybrid (v8) was `@champion` for a few hours, then the user switched back to v5. v8 stays `@challenger`, verified 7/7. The false-alarm budget, id-gap rule and gauge flag were left as they are (see ARCHITECTURE.md).
- [ ] **Stretch:** sea-level-rise scenarios (re-score with tide offsets); upstream-catchment rainfall; radar and lightning features. (The separate flash-flood model for High is the ordinal model's flood rung, 23 Oct.)

### Analytics and BI

`resources/app.yml` and `src/app/` hold the FloodSense app (21 Oct; see [The app](#the-app)). Since 25 Oct, `resources/dashboard.yml` (with `src/dashboards/scoreboard.lvdash.json`) and `resources/genie.yml` hold the agencies' AI/BI scoreboard and Genie space. Both read the tables `flood-scoreboard` publishes, and the app's "For agencies" view shows both (see [For agencies](#for-agencies-the-scoreboard-and-genie)).

- [ ] FloodSense app (a Databricks App, for residents)
    - [ ] Choose the framework (AppKit, or Python if the team prefers) and deploy a stub to the final demo workspace early, as the guide advises. **W1** React + FastAPI, built and run locally against dev on 21 Oct; not yet deployed.
    - [x] Map of the 47 areas, coloured by tier from the newest `ml.predictions` (the champion pair, else the rule).
    - [x] Area panel: tier, `p_alert` as plain odds, the reason, the last hour's rain, the tide, and actions for drivers, residents and shop owners (`frontend/src/lib/content.ts`).
    - [x] Live mode: rechecks every minute, shows when the forecast was made, and says so when it is over 20 minutes old.
    - [x] Replay mode: storm days after the training data, scored tick by tick by the @champion model in the app, with PUB's alerts and a did-it-warn-first table.
    - [x] Scoreboard view ("How reliable is it?", from the champion's MLflow run), plus the line that FloodSense is an earlier layer on top of PUB's alerts, not a replacement.
    - [x] Works on a phone (checked at 375 px).
- [ ] AI/BI dashboard (for agencies): "FloodSense scoreboard", deployed to dev and embedded in the app on 25 Oct. Four pages: Scoreboard, Every flash flood, Warnings by area, PUB alerts and flood-prone land.
    - [x] Flood-prone trend panel (`gold.flood_prone_trend`, 2022–2025). The 36 listed locations aren't transcribed yet, so the page maps PUB's alert locations instead. **W1**
    - [x] PUB's re-measured warning time against the 30-minute target: PUB's own warning before every flash flood in the archive (`ml.scoreboard_floods.pub_warning_min`), beside FloodSense's lead and the rule's. **W1**
    - [x] Scoreboard for the model, the rule and PUB: share of PUB events and flash floods warned before PUB, lead time, PUB's own warning, and false alarms per area-month, with their 95% CIs. High precision is left out: the tiers are fixed chances of any PUB alert, so Elevated is judged as a warning, not as a flood forecast. **W2**
    - [ ] Pipeline health: freshness per source, expectation pass rates, and geocoding misses awaiting review. **W2**
- [x] Genie space **W2**: "FloodSense for agencies", deployed to dev on 25 Oct, and asked from the app.
    - [x] The scoreboard tables, `pub_alerts` (silver.area_alerts), `flood_prone_trend` and `live_risk` (ml.predictions), with column comments, instructions, six example queries and five sample questions.
    - [x] Test the demo questions in advance: all five sample questions answered right through the Conversation API on 25 Oct, in 14–22 s each. "Which areas had the most FloodSense false alarms after training?" gives Serangoon 90, Jurong West 76, Toa Payoh 60. Screenshots of the live answers are still to take.
- [ ] **Stretch:** Telegram push alerts, or a hand-off to PUB's channel.

### The app

`src/app/` is a Databricks App: a React front end (`frontend/`, Vite + TypeScript) served by FastAPI (`backend/`). It reads, through the SQL warehouse:

- `ml.predictions`: every area's warning in force and its newest rating (the pair over the rule when both scored it), and an area's last hours;
- `silver.rainfall_5min` with `gold.area_rain_stations`: the wettest gauge within 5 km over the last hour;
- `gold.tide_levels`: the tide now and the next high water;
- `bronze.ingest_heartbeat`: when the feed last fetched.

The track record comes from the @champion version's MLflow run (out-of-fold metrics, the rule's, and the without-tide comparison). Replays download the @champion model from Unity Catalog once and score a past day's `gold.area_features` (47 areas × 288 ticks, about 40 s the first time, cached after). They're limited to days from `contract.TEST_START` (20 Dec 2025) on, which the model never trained on. The map's areas are Voronoi cells around NEA's label points clipped to the Master Plan 2019 land-use outline, built once by `src/app/scripts/build_area_map.py` into `frontend/public/areas.geojson`. Address search goes through OneMap; "Use my location" works without it.

Run it locally (Python 3.11, as the Apps runtime):

```bash
(cd src/app/frontend && npm ci && npm run build)
pip install -r src/app/requirements.txt
DATABRICKS_CONFIG_PROFILE=flood_dev DATABRICKS_WAREHOUSE_ID=<warehouse id> FLOODSENSE_SCHEMA_PREFIX=dev_<you>_ \
  uvicorn backend.main:app --app-dir src/app --port 8000
```

**A warning lasts an hour (26 Oct).** Every Moderate or Elevated rating is a warning for the next hour, as the model was trained. So an area stays warned until an hour after the rating that warned it, and ratings at most an hour apart are one warning. Elevated has priority: while any Elevated rating is under an hour old the area shows Elevated, so Moderate can follow only an hour after the last Elevated rating. The rule lives in four places that must agree: the live map (`backend/main.py`, `warning_in_force`), the replays (`frontend/src/lib/warnings.ts`), the Telegram posts (`ml/broadcast.py`, `in_force`) and the `live_risk` view (`ml/publish.py`). The scoreboard already counted warnings this way (`scoreboard.warning_episodes`), so its numbers didn't change.

A replay's "warned N min ahead" now runs from the first rating of the warning in force at PUB's first alert. A rating at the same tick as PUB's alert no longer counts. Over the 92 area-days with a PUB alert from 20 Dec 2025 to 24 Oct 2026 (final_2), FloodSense warned first on 46. When it did, its head start averaged 18.6 min (median 15.5), against 14.0 min on 49 area-days by the old count (an unbroken run of warned ticks). Counting misses as 0, the average is 9.3 min (was 7.4). Per PUB event, as the scoreboard counts: 17.0 min on 29 of 60 after training, and 26.3 min on 144 of 323 out of fold.

Every view but "For agencies" ends with **Ask a question** (`components/AskPanel.tsx`), the same Genie space with questions that fit the view (`ASK_QUESTIONS` in `lib/content.ts`, each checked on 26 Oct). The Now view also takes sign-ups for **warnings by email or text** (`components/Subscribe.tsx`, `/api/subscriptions`). They go to `<app schema>.warning_subscriptions`, a schema of its own because it holds personal data, and the app shows addresses back masked. Warnings wherever the visitor is go through the Telegram bot: the Now view links to it (`/api/follow`, variable `warning_bot`), or shows the option switched off, as coming soon, when the app has no bot. See [Warnings wherever you are](#warnings-wherever-you-are-the-telegram-bot).

Replays read every feature the champion reads (`Champion.features`, from the model's own contract). Until 23 Oct they read a fixed list that left out the nearby and upwind rain and the weather, so a replay scored storms without the pair's second-strongest input. The track record shows the out-of-fold flash floods too (`oof_floods_*`), when the champion's run logged them.

**Free Edition: the app and the live loop can't run at once.** The app reads through the SQL warehouse, and a pipeline update fails with `RESOURCE_EXHAUSTED` while the warehouse runs, so every `flood-live` run's `refresh_pipeline` fails while anyone has the app open (the warehouse stays up until it idles out). To show the app: run `flood-live` once with the warehouse stopped, then start the app; it shows that tick until the next run. A demo that must stay live needs a paid workspace, or a run of `flood-live` between viewing sessions.

Deploy: build the front end first (the bundle syncs `frontend/dist`, which git ignores), then `databricks bundle deploy -t dev -p flood_dev` and `databricks bundle run floodsense_app -t dev -p flood_dev`. The bundle gives the app's service principal the warehouse (CAN_USE) and the experiment (CAN_READ). Unity Catalog access is granted separately, once, with the principal's application id:

```sql
GRANT USE CATALOG ON CATALOG flood_dev TO `<app principal>`;              -- by the catalog's owner
GRANT USE SCHEMA, SELECT ON SCHEMA flood_dev.dev_<you>_ml TO `<app principal>`;   -- and _silver, _gold, _bronze
GRANT EXECUTE ON MODEL flood_dev.dev_<you>_ml.dev_<you>_floodsense TO `<app principal>`;
```

### Warnings out: Telegram, email and text

After each run, `flood-live`'s score task sends what changed (`ml/broadcast.py`). An area gets a post when its warning starts, steps up to Elevated or steps down to Moderate, and none while it holds. One Telegram message per run goes to `@daisi_floodsense` (variable `warning_channel`). Each subscriber gets one email or text per run for their areas, and a welcome message after signing up. `ml.warning_posts` logs every post, so a re-run never posts twice. Nothing is sent for a tick more than 20 minutes old. A failure is logged and never fails the scoring.

One-time setup, by the user:

1. Telegram: create a bot with @BotFather, and add it to the channel as an administrator allowed to post. Then store its token:

    ```bash
    databricks secrets put-secret floodsense telegram-bot-token -p flood_dev
    ```

2. Email: the same Gmail account as the parser alert. `parser-alert-smtp-user` must be the full address, and `parser-alert-smtp-password` a 16-letter app password.
3. Texts: a Twilio account, in `twilio-account-sid`, `twilio-auth-token` and `twilio-from-number`. Without them, text sign-ups are kept but nothing is sent, and the app says so.
4. After the first deploy, give the app's service principal the subscriptions table. The first `flood-live` run creates the table:

    ```sql
    GRANT USE SCHEMA ON SCHEMA flood_dev.dev_<you>_app TO `<app principal>`;
    GRANT SELECT, MODIFY ON TABLE flood_dev.dev_<you>_app.warning_subscriptions TO `<app principal>`;
    ```

Try the senders without a storm (it reads the secrets as your profile):

```bash
uv run python -m daisi_team_master_corgis_floodsense.ml.broadcast --profile flood_dev --test-telegram --test-email you@example.com
```

### Warnings wherever you are: the Telegram bot

A web page can't follow a phone once it's closed, so the bot does (`ml/follow.py`). The Now view's **Follow me on Telegram** opens `t.me/<warning_bot>?start=follow`. The person taps Start and shares their live location (📎, Location, Share My Live Location), or sends a pin to follow one place. Telegram sends each new position as an edit of that message.

Nothing outside can reach `flood-live`, so the bot fetches its own messages. After `broadcast.run`, each run's score task reads what Telegram kept since the last run (`getUpdates`, kept 24 hours), answers it, and confirms it. So the bot answers within one run, and only while `flood-live` runs: unpause it after each dev deploy.

- **Only the area is kept.** `<app schema>.telegram_followers` holds each chat's forecast area, never coordinates. The area is the one the app's map draws the position in (`public/areas.geojson`, clipped to land). A phone in Johor or out at sea is in none, and is told so.
- **When it writes.** A follower hears when a warning starts, steps up or steps down in their area, or when they move into an area under a warning they haven't heard about. It says nothing while a warning holds or lapses, or for a tick more than 20 minutes old. `/status` says what it follows; `/stop`, the end of the live location, or blocking the bot stops it.
- **One deployment reads the bot.** Telegram hands each message to whichever caller confirms it first, so two deployments reading one bot would split its followers. `follow_telegram` is `"false"` unless set, and exactly one deployment sets it, in `.databricks/bundle/dev/variable-overrides.json`:

    ```json
    { "follow_telegram": "true" }
    ```

- **No webhook.** `getUpdates` fails while the bot has one. The bot is the same as the channel's (`telegram-bot-token`), so nothing else needs setting up.

### For agencies: the scoreboard and Genie

Built on 25 Oct. There are three pieces, all running on the one SQL warehouse:

1. **`flood-scoreboard`** (`resources/scoreboard.job.yml`, `ml/publish.py`). It replaces four tables in the ml schema and puts three views beside them, so the dashboard and Genie read one schema. Run it whenever `@champion` moves; it takes about 4 minutes. On Free Edition it runs while the warehouse is up: only pipelines clash with the warehouse.

    ```bash
    databricks bundle run scoreboard -t dev -p flood_dev
    ```

    - `scoreboard`: a row per period, model and warning level (FloodSense from Moderate, FloodSense at Elevated, and the rule). The development rows are the champion training run's own out-of-fold metrics, read back from MLflow. The after-training rows come from scoring every area-tick of `gold.area_features` from 20 Dec 2025 with the champion and its rule.
    - `scoreboard_floods`: every flash flood, with PUB's own warning, FloodSense's lead and the rule's. FloodSense's lead is out of fold before 20 Dec 2025, and the live version's after.
    - `scoreboard_events` and `scoreboard_warnings`: every PUB event, and every warning after training with whether a PUB alert followed it.
    - The views `pub_alerts`, `flood_prone_trend` and `live_risk`. Times are SGT wall-clock times (`TIMESTAMP_NTZ`), and tiers use the app's names (Moderate, Elevated).
2. **The dashboard**, "FloodSense scoreboard" (`resources/dashboard.yml`). It is published with the deployer's credentials (`embed_credentials`), so a viewer needs only CAN_RUN on it, which the `users` group has. Its queries name tables bare, and `dataset_catalog` and `dataset_schema` point them at the ml schema.
3. **The Genie space**, "FloodSense for agencies" (`resources/genie.yml`). It is defined inline in the YAML rather than in a `.geniespace.json` file, so its table names follow the target. Its six example queries were checked on the warehouse. One of them teaches it to keep 2022 in the flood-prone trend: 2022's change is null, and without the example Genie dropped the baseline year.

The app's "For agencies" view (`frontend/src/views/AgenciesView.tsx`) shows both:

- **Scoreboard** embeds the published dashboard. `/api/agencies` gives its address, `https://<host>/embed/dashboardsv3/<id>?o=<workspace id>`. The dashboard shows only when the viewer is signed in to Databricks in the same browser with third-party cookies allowed (Chrome's default), and only on a domain the workspace approves (Settings > Security > Embed dashboards). The view's "Open the scoreboard in Databricks" link is the fallback. A workspace admin approves the app's host (bare, as `floodsense-<you>-<workspace id>.aws.databricksapps.com`) once, in the UI or with:

    ```bash
    databricks settings aibi-dashboard-embedding-approved-domains update -p flood_dev --json \
      '{"allow_missing": true, "field_mask": "aibi_dashboard_embedding_approved_domains.approved_domains",
        "setting": {"aibi_dashboard_embedding_approved_domains": {"approved_domains": ["<app host>"]}}}'
    ```
- **Ask a question** sends questions to the Genie space through the Conversation API, as the app's service principal. The principal has CAN_RUN from the app's `genie-space` resource, and reads the tables through its SELECT on the ml schema. Genie takes 10–30 s, longer when the warehouse starts cold. So `POST /api/ask` starts a message, and the browser polls `GET /api/ask/{conversation}/{message}` every 1.5 s. A finished answer is cached: the same first question asked again within 6 hours gets it at once, so the demo's questions answer instantly once someone has asked them. The five sample questions live in both `frontend/src/lib/content.ts` and `genie.yml`; change them together.

After training (20 Dec 2025 to 24 Oct 2026, published 25 Oct):

- FloodSense from Moderate warned before PUB for 48% of 60 judged PUB events (95% CI 36–61%), 8.2 minutes ahead on average, at 1.34 false alarms per area-month.
- At Elevated alone, FloodSense warned first for 18%, at 0.64 false alarms per area-month.
- The rule warned first for 17%, at 0.67.
- Of the flash floods, FloodSense warned of 2 of 5 before PUB, and the rule of 3 of 5.

The 2026 floods were looked at while versions were chosen (verify's `floods.json`), so this is the record after training, not an untouched test.

A view keeps the session settings it was created in. The job's session is SGT, and `convert_timezone(current_timezone(), ...)` in a view mixed the creator's time zone with the reader's: it shifted `live_risk` by 16 hours. The views now turn epoch seconds into SGT with Singapore's fixed +08:00, which needs no time zone at all.

### Governance (Stretch)

- [ ] Replace the "Placeholder" comments in `resources/uc.yml`, and give silver, gold and ml columns the descriptions in `contract.py`.
- [ ] Lineage screenshot for the deck: landing volume → bronze → silver → gold → model → `ml.predictions` → app and dashboard.
- [ ] Grants: the app's service principal and Genie users read `gold` and `ml` only. Restrict the `floodsense` secret scope.
- [ ] Model governance: a UC model description, tags (training window, test metrics), the training table logged as an MLflow input, and `@champion` moved only after a scoreboard check.
- [ ] Prod target: create the deploy service principal and run its grant in `setup/bootstrap.sql`. Remove `trigger_pause_status: PAUSED` once ingestion runs there.
- [ ] Attribution in table comments and the README: data.gov.sg (Singapore Open Data Licence), NEA, PUB, URA, SLA, UHSLC, OneMap, Copernicus (if GLO-30 is used) and TypeSafe AI's Jev (through Databricks).

### Demo Day (27 Oct, 4–7pm SGT)

- [ ] Ask our mentor what the Round 2 submission must contain and when it's due. The guide only says "before Demo Day".
- [ ] Commit and push the ingestion and ML code, which was still untracked on 13 Oct. Update the README (results, architecture, how to run) and the `pyproject.toml` description. Make the repo accessible to the judges.
- [ ] Choose the final demo workspace and target. Dev names carry a `[dev …]` prefix, and dev deploys leave schedules paused.
- [ ] Quota:
    - [ ] Keep `live_schedule` at 15 minutes outside demo windows.
    - [ ] Run no backfills or full refreshes on 26–27 Oct; going over the daily quota stops compute for the rest of the day.
- [ ] Restart the app within 24 hours of the demo (Free Edition stops apps after 24 hours). Check that the SQL warehouse, pipeline and data are ready, and walk the main journey from a fresh browser session.
- [ ] Keep recordings or screenshots of the steps most likely to fail: live mode, Genie and the replay.
- [ ] No tokens, keys or personal data on screen. Tell the judges what isn't live: the precomputed replay, labels read by Jev or fixed by hand, and prototype features.
- [ ] Rehearse the story and the presenter handoffs by 25 Oct: problem (13.3 min) → users → architecture (batch backfill for training, streaming ticks for inference) → replay → live → scoreboard → Genie → impact (minutes gained over PUB per 2026 flood, with CIs). Prepare answers on the 8-flood test, geocoding misses, and why NEA's 47 areas.

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
| Response | Type | Live Dataset | Historical Equivalent | Time Range | Time Interval | Live Source | Historical Source | Remarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flash Floods | Boolean | PUB Flood Alerts (Live API) | Same | May 22, 2025 to Present | Minutes | [PUB](https://data.gov.sg/datasets/d_f1404e08587ce555b9ea3f565e2eb9a3/view) | NA | [See Appendix](#flash-floods-live-api) |
| Flash Floods & Water Level Sensor alerts (inferred to trigger at 90% water level) | Boolean | PUB Flood Alerts (Telegram) | Same | November 17, 2021 to Present, with gaps | 1 Minute for Training, 5 Minutes for Live | [Telegram PUB Flood Alerts](https://t.me/s/pubfloodalerts?before=1632) | Wayback Machine captures of `t.me/sgflood` (until 14 Nov 2023) and `t.me/pubfloodalerts` | [See Appendix](#flash-floods--water-level-sensor-alerts) |



#### Features
| Feature | Type | Live Dataset | Historical Equivalent | Time Range | Time Interval | Live Source | Historical Source | Remarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rainfall | Numerical | NEA Rainfall across Singapore (2016 to Present) | NEA Historical Rainfall (2016 to 2024) | 2016 to Present | 5 minutes | [NEA](https://data.gov.sg/datasets/d_6580738cdd7db79374ed3152159fbd69/view) | [NEA](https://data.gov.sg/collections/2279/view) | [See Appendix](#rainfall) |
| 2-Hour Weather Forecast | Categorical ([See Appendix](#2-hour-weather-forecast-possible-categories)) | NEA 2-Hour Weather Forecast (2016 to Present) | NEA Historical 2-Hour Weather Forecast (2016 to 2024) | 2016 to Present | 30 minutes | [NEA](https://data.gov.sg/datasets/d_3f9e064e25005b0e42969944ccaf2e7a/view) | [NEA](https://data.gov.sg/collections/2179/view) | [See Appendix](#2-hour-weather-forecast-time-interval) |
| Air Temperature | Numerical | NEA Air Temperature across Singapore (May 2016 to Present) | NEA Historical Air Temperature across Singapore (2016 to 2024) | May 2016 to Present | 1 minute | [NEA](https://data.gov.sg/datasets/d_66b77726bbae1b33f218db60ff5861f0/view) | [NEA](https://data.gov.sg/collections/2246/view) | [See Appendix](#air-temperature) |
| Relative Humidity | Numerical | NEA Relative Humidity across Singapore (2016 to Present) | NEA Historical Relative Humidity across Singapore (2016 to 2024) | May 2016 to Present | 1 minute | [NEA](https://data.gov.sg/datasets/d_2d3b0c4da128a9a59efca806441e1429/view) | [NEA](https://data.gov.sg/collections/2278/view) | [See Appendix](#relative-humidity) |
| Wind Direction | Numerical (circular) | NEA Wind Direction across Singapore (2016 to Present) | NEA Historical Wind Direction across Singapore (2016 to 2024) | Nov 2016 to Present | 1 minute | [NEA](https://data.gov.sg/datasets/d_534cf203023b51f51f879145ccc56ff9/view) | [NEA](https://data.gov.sg/collections/2281/view) | [See Appendix](#wind-direction) |
| Speed | Numerical | NEA Wind Speed across Singapore (2016 to Present) | NEA Historical Wind Speed across Singapore (2016 to 2024) | Dec 2016 to Present | 1 minute | [NEA](https://data.gov.sg/datasets/d_7677738484067741bf3b56ab5d69c7e9/view) | [NEA](https://data.gov.sg/collections/2280/view) | [See Appendix](#wind-speed) |
| Tide levels | Numerical | NEA Tide Timings | UHSLC Tanjong Pagar (imputed via utide) | 2016 to Present (UHSLC span unconfirmed) | Live: high/low times (about 4 a day); Historical: hourly | [NEA](https://www.nea.gov.sg/weather/tide-timings) | [UHSLC](https://uhslc.soest.hawaii.edu/data/download/fd#uh699) | [See Appendix](#tide-levels) |
| URA Master Plan | Categorical (land-use zoning polygons) | URA Master Plan 2025 Land Use Layer | URA Master Plan 2019 Land Use layer | Nov 2019 to Present | Static (new Master Plan roughly every 5 years) | [URA](https://data.gov.sg/datasets/d_a8c3546b26712e35021f3a681d0353ae/view) | [URA](https://data.gov.sg/datasets/d_90d86daa5bfaa371668b84fa5f01424f/view) | [See Appendix](#ura-master-plan) |
| Drainang Catchment Maps | Categorical (catchment polygons) | NA (static) | PUB Drainage Catchment Map + Drainage Boundary (Northern & Southern Coast) Map | Static | Static | NA | [PUB](https://www.pub.gov.sg/Professionals/Requirements/Qualified-Persons/Catchment-Maps) | [See Appendix](#drainage-catchment-maps) |
| Elevation | Numerical (raster, m) | NA (static) | SLA elevation; fallback: Copernicus GLO-30 DEM | Static (GLO-30 built from 2011–2015 TanDEM-X data) | Static | NA | [SLA](https://data.gov.sg/datasets?topics=environment&resultId=d_10480c0b59e65663dfae1028ff4aa8bb); [Copernicus](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) | [See Appendix](#elevation) |

#### Validation

| Variable | Type | Live Dataset | Historical Equivalent | Time Range | Time Interval | Live Source | Historical Source | Remarks |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| List of Flood Prone Areas | Categorical (36 named locations); Numerical (total ha) | NA | PUB List of Flood Prone Areas (Apr 2025); PUB Flood Prone Areas (annual, 2022–2025) | 2022 to 2025 | Annual | NA | [PUB list](https://pub.gov.sg/-/media/PUB/PDF/Flood-Management/List-of-Flood-Prone-Areas-as-at-Apr-2025.pdf), [PUB hectares](https://data.gov.sg/datasets/d_c4aed98f1533eb3a66f65dbb1a30da46/view) | [See Appendix](#list-of-flood-prone-areas) |
| PUB Water Level Sensors Locations | Geospatial (points) | NA | PUB Water Level Sensors | Snapshot (last updated 13 Apr 2026) | Refreshed half-yearly | NA | [PUB](https://data.gov.sg/collections/1541/view) | [See Appendix](#pub-water-level-sensors-locations) |

## Ingestion (bronze)

Every input lands raw in the UC volume `/Volumes/<catalog>/<bronze schema>/landing/<name>/…`. In the `flood-etl` pipeline, Auto Loader streams each folder unchanged into `bronze.<name>`. The folder name is the table name. `tests/test_bronze_contract.py` checks that the two sides agree.

| Landing folder → bronze table | Source | Cadence |
| --- | --- | --- |
| `nea_rainfall`, `nea_air_temperature`, `nea_relative_humidity`, `nea_wind_speed`, `nea_wind_direction`, `nea_two_hr_forecast`, `nea_lightning`, `nea_radar_70km`, `pub_flood_alerts` | data.gov.sg v2 real-time API | Every tick: newest page (25 records). Daily: the full past days |
| `pub_telegram_posts` | t.me/s/pubfloodalerts, PUB's channel (90% warnings, flash floods, NEA heavy rain); web.archive.org for the posts PUB deleted | Every tick (new posts only), plus a one-off channel scrape and a one-off recovery of deleted posts |
| `nea_tide_timings` | NEA tide-timings page (this month and next) | Once a day |
| `*_hist` (6 tables) | data.gov.sg yearly "Historical …" CSVs | One-off, for every published year from 2021 |
| `uhslc_tanjong_pagar_hourly` | UHSLC research-quality hourly sea level, 1984–2023 | One-off |
| `static_*` (5 tables) | URA land use 2019/2025, PUB water-level sensors, flood-prone area (ha), SLA benchmarks | When data.gov.sg updates them |
| `ingest_heartbeat` | One line per tick: per-source status, record counts, latency | Every tick |

Radar PNGs go to `landing/radar_png/70km/YYYY/MM/DD/`, with no table. Their presigned URLs expire after about 25 min, so every tick saves the new frames. Fetcher cursors live in `landing/_state/`.

JSON lines are parsed into a VARIANT `payload` column, so query them like `payload:body.data.readings`. CSV columns stay as text. Typing, dedup and parsing belong in silver.

### Where things run

The egress probe confirmed on 2026-10-10 that Free Edition serverless can reach data.gov.sg, its S3 downloads, t.me, nea.gov.sg and UHSLC. So the fetchers run as jobs:

- **`flood-live`** (`resources/live.job.yml`): every `${var.live_schedule}` (default 5 min), runs `fetch`, `refresh_pipeline`, then `score` (the newest ticks into `ml.predictions`; see [Staying live](#staying-live)).
- **`flood-watchdog`** (`resources/watchdog.job.yml`): hourly, emails the deployer when the live feed or the predictions go stale.
- **`flood-daily`** (`resources/daily.job.yml`): at 00:30 SGT, re-pulls the last two days in full, so gaps in the live feed get filled.
- **`flood-backfill`** (`resources/backfill.job.yml`, manual): static layers, Telegram archive, UHSLC and yearly CSVs, then a pipeline refresh.

**The long API backfill runs on a laptop, not serverless.** That is 2025-01-01 → yesterday, about 190k requests. It mostly waits on rate limits, and on serverless that waiting still burns the Free Edition daily quota. Expect about 3 days with a dev API key, about 1 day with a prod key, and about 6 days without a key. Sources run in priority order, so labels, rainfall and forecast land in roughly the first 8 hours with a dev key.

**So does the web archive recovery.** `backfill.py telegram-wayback` reads about 90 captures from web.archive.org, 4 s apart, in about 12 minutes. The egress probe doesn't check web.archive.org, and on serverless the waiting would burn quota.

If the probe ever starts failing (`uv run python src/ingestion/probe_egress.py`, run as a one-off job), switch to local mode:

```bash
uv run python src/ingestion/fetch_live.py --landing /Volumes/flood_dev/dev_<you>_bronze/landing \
    --profile flood_dev --every 5m --daily-repair --trigger-job <flood-live job id>
```

In local mode, also remove the `fetch` task and the schedule from `live.job.yml`.

### Staying live

What "live" means here: every few minutes, `ml.predictions` gains the newest tick for all 47 areas, read by the app and the dashboard. Until 21 Oct the chain stopped at silver: `flood-live` fetched and refreshed `flood-etl`, but nothing scored, and gold refreshed only at night. Now:

| Job | When | What keeps it live |
|---|---|---|
| `flood-live` | every `live_schedule` (5 min) | `fetch` (240 s timeout) → `refresh_pipeline` (600 s) → `score` (300 s, one retry). Overlapping runs are skipped, not queued, and the next run catches up: fetch reads the newest 25 records of each source, and score scores every tick since its last, at most 2 hours back. |
| `flood-daily` | 00:30 SGT | re-pulls the last two days in full, refreshes gold (`area_rain_stations`, `tide_levels`, which score reads). Emails the deployer on failure. |
| `flood-watchdog` | hourly at :07 | fails, and emails the deployer, when the newest tick, core-source fetch, rain reading or prediction is older than 20/20/30/30 min (`freshness.py`). |

- **`score`** (`ml/live.py`, `src/ml/score_live.py`) computes the newest ticks' features from silver with gold's own functions (`daisi_team_master_corgis_floodsense/features.py`, which `gold/features.py` now calls too), so a live prediction reads exactly what the models learnt from. Checked locally on 21 Oct: over a span of 15 ticks, its rows equal `gold.area_features`' row for row. It scores only ticks whose rain window has reached silver (newest reading + 10 min), so a failed refresh scores nothing rather than a false all-clear, and each tick once. It uses `@champion`, else the rule from the newest `flood-train` run (the `rule` parameter); a pair saved before the tide columns still loads, as a pair records its inputs.
- **A fetch fails only when a core source fails** (`sources.CORE_SOURCES`: rainfall and the 2-hour forecast, what every tick's features need). Radar, lightning, Telegram and the rest still show `ok: false` in the heartbeat, and `flood-daily` repairs their days, but they no longer mark every `flood-live` run failed.
- **Turn it on after each deploy.** Dev deploys pause every schedule (`mode: development`), so a redeploy stops the feed until it's unpaused. Unpause `flood-live`, `flood-daily` and `flood-watchdog` in the Jobs UI, or:

  ```bash
  for name in flood-live flood-daily flood-watchdog; do
    id=$(databricks jobs list -p flood_dev -o json | jq -r --arg n "$name" '.[] | select(.settings.name | endswith($n)) | .job_id')
    databricks jobs get $id -p flood_dev -o json | jq '{job_id, new_settings: {schedule: (.settings.schedule + {pause_status: "UNPAUSED"})}}' \
      | databricks jobs update --json @/dev/stdin -p flood_dev
  done
  ```

- **Quota.** Each `flood-live` run starts serverless compute three times (job tasks and the pipeline). If the Free Edition daily quota runs out, everything stops until the next day, and `flood-watchdog`'s email is the sign. Outside demo windows run `live_schedule: "0 0/15 * * * ?"`; for the demo, 5 minutes. A `flood-etl` update takes about 5 minutes (14 Oct), so at a 5-minute cadence about every other run is skipped: predictions arrive roughly every 10 minutes either way.
- **Read times as epoch seconds.** `collect()` returns a TIMESTAMP in the driver's zone (UTC on serverless) whatever the session time zone, so the first live run (21 Oct) read 16:10 SGT as 08:10 and scored a tick 8 hours old. `freshness.sgt` converts; `toPandas()` does follow the session time zone.
- **First live run, 21 Oct:** the 16:25 SGT tick scored for all 47 areas with features, 4.8 min behind the clock, with `dev_eugenehengruizhe_floodsense v1` (`@champion`); `flood-watchdog` reported live.
- **Check it:**

  ```sql
  SET TIME ZONE 'Asia/Singapore';
  SELECT model, max(ts) AS newest_tick, count(DISTINCT area) AS areas, timestampdiff(MINUTE, max(ts), current_timestamp()) AS lag_min
  FROM flood_dev.dev_<you>_ml.predictions WHERE ts > current_timestamp() - INTERVAL 1 DAY GROUP BY model;
  ```

  `databricks bundle run watchdog -t dev -p flood_dev` runs the same check the hourly job does.

### One-time setup

1. **data.gov.sg API key.** Without one everything still works, at the lowest rate limit: 6 real-time calls per 10 s, against 12 with a dev key and 30 with a prod key. So the long backfill takes about twice as long.
   1. **Get one.** Log in at [data.gov.sg](https://data.gov.sg) (top right; sign up with your email if you're new), click **Create API Key**, and choose **Developer** (for testing, lower limits) or **Production** (higher limits; describe what it's for). Copy it straight away: it's shown only once. Keys expire; the dashboard's API Keys section shows when.
   2. **Store it** in the workspace's `floodsense` secret scope. Jobs and laptop runs both read it there:

      ```bash
      databricks secrets create-scope floodsense -p flood_dev                 # once per workspace; skip it if the scope exists
      databricks secrets put-secret floodsense datagovsg-api-key -p flood_dev  # prompts for the key, which keeps it out of shell history
      ```

      Storing it again replaces the old value.
   3. **Check it** while nothing else is using the key:

      ```bash
      uv run python src/ingestion/check_key.py --profile flood_dev
      ```

      It should report 12 calls per 10 s (dev) or 30 (prod). data.gov.sg answers a key it doesn't recognise (mistyped, expired) as if there were none, so 6 means the key needs fixing. Nothing else would tell you.
   4. **A prod key:** add `--key-tier prod` to runs, or `export DATAGOVSG_KEY_TIER=prod`. Runs assume dev otherwise.

   How runs find the key: `$DATAGOVSG_API_KEY` if it is set (it overrides the scope, for a one-off key), else `datagovsg-api-key` in the secret scope (`--secret-scope`, default `floodsense`). Jobs read it through dbutils; laptop runs read it through the workspace API with their `--profile`. The log names the source and the tier (`data.gov.sg API key from secret scope 'floodsense'`, `rate tier: dev`), never the key. A run that data.gov.sg rate-limits below its key's limit warns once and points at `check_key.py`.
2. **Databricks CLI on PATH, for local runs.** The `flood_dev` profile uses `auth_type = databricks-cli`, so the SDK shells out to `databricks`. Either install the standalone CLI, or set `DATABRICKS_CLI_PATH` to the binary inside the VS Code extension.
3. **Parser alerts: the sending account and Jev.** flood-live emails the developers when PUB posts something no pattern reads ([When no pattern reads a post](#when-no-pattern-reads-a-post-jev)), and asks Jev to read it.
   1. **The sending account.** Make a Gmail app password for the account that sends (Google Account → Security → 2-Step Verification → App passwords; it needs 2-Step Verification on), then store both in the scope:

      ```bash
      databricks secrets put-secret floodsense parser-alert-smtp-user -p flood_dev      # the Gmail address
      databricks secrets put-secret floodsense parser-alert-smtp-password -p flood_dev  # the 16-character app password
      ```

      Without them a fallback still lands Jev's call and logs the whole email as a `WARNING`, but sends nothing. Another provider works with `--smtp-host` and `--smtp-port` (implicit TLS).
   2. **Jev (off).** `system.ai.typesafeai-jev-1-14-0` is served only with the account-level preview **Externally-hosted TypeSafe AI Models** (account console → Previews), and Databricks charges for its use, so it stays off: every fallback is a Jev error, emailed, with the post's label windows missing. Turning the preview on is all it takes to use Jev; the caller needs `EXECUTE` on the model service, which `system.ai` grants to account users by default.
   3. **Send a test alert** through the live fallback, on a made-up post in a new format (it asks Jev once, a free 404 while the preview is off, landed under `telegram_jev_parses/test/`; the email is marked TEST):

      ```bash
      uv run python src/ingestion/check_parser_alert.py --landing /Volumes/flood_dev/dev_<you>_bronze/landing --profile flood_dev \
          --email-to you@example.com
      ```

      It fails only when the email can't be sent (it names the secrets to fix) or when Jev answers but misreads the post (a warning at Jalan Boon Lay (Bus stop B21339) at 15:27 and Upp Paya Lebar Rd at 15:31).
4. **Local Python environment.** `uv sync` builds `.venv` to match serverless environment 6, which `flood-train` runs on: Python 3.12, that environment's library versions (the `[tool.uv]` pins in `pyproject.toml`, written by `databricks environments setup-local`), and databricks-connect 19.1, which runs Spark code on serverless from a laptop (`DatabricksSession.builder.profile("flood_dev").serverless(True).getOrCreate()`). pyspark comes with databricks-connect; a standalone pyspark can't share its environment. When `flood-train` moves to a newer environment, refresh the pins: `databricks environments setup-local --job-task <flood-train job id>.train -p flood_dev`, after a `--dry-run`. The ingestion jobs still run environment 2 (Python 3.11), which `ruff` checks against.

### Commands

Local runs need `--profile flood_dev`; `--landing` can also come from `$FLOODSENSE_LANDING`.

```bash
L=/Volumes/flood_dev/dev_<you>_bronze/landing
uv run python src/ingestion/fetch_live.py   --landing $L --profile flood_dev          # one tick
uv run python src/ingestion/fetch_static.py --landing $L --profile flood_dev
uv run python src/ingestion/backfill.py telegram --landing $L --profile flood_dev
uv run python src/ingestion/backfill.py telegram-wayback --landing $L --profile flood_dev   # posts PUB deleted, ~12 min
uv run python src/ingestion/backfill.py uhslc    --landing $L --profile flood_dev
uv run python src/ingestion/backfill.py csv      --landing $L --profile flood_dev --sources nea_rainfall --years 2024
# the long one, in the background; it resumes where it stopped, so Ctrl-C or a reboot is fine
nohup uv run python src/ingestion/backfill.py api-days --landing $L --profile flood_dev > ~/floodsense-api-days.log 2>&1 &
```

For a local smoke test, use `--landing ./.landing-dev --sink local`; that folder is gitignored. To run on Databricks without a schedule, use `databricks bundle run backfill -t dev`, or a one-off `databricks jobs submit` with the same script and arguments.

Every backfill step is idempotent:

- **csv:** skips years already landed and logs years data.gov.sg hasn't published yet. As of 2026-10-10 that is 2025 and 2026, so those days come from `api-days`. Re-run it when a new yearly file appears. If a file is published outside its collection, add it with `--dataset-id nea_rainfall:2025:d_…`.
- **api-days:** writes one gzipped file per source-day and skips days that already exist. A day without data still gets its file (holding the 404 or the empty page), so it isn't retried. Use `--max-runtime 5h` to stop cleanly, `--last-days 2` for the daily repair, and `--sources …`/`--start`/`--end` to target a range.
- **telegram-wayback:** writes one file per web archive capture it reads, even an empty one, and rebuilds what it knows from those files, so a re-run reads only captures that can still add a post. `--until` (default 2025-03-31) is the last capture day it reads; later captures repeat what `telegram` lands.

### Checking it

```sql
-- live feed freshness, per source
SELECT r.value:source::string AS source, max(h.fetched_at) AS last_tick,
       max_by(r.value:ok::boolean, h.fetched_at) AS last_ok, max_by(r.value:records::int, h.fetched_at) AS records
FROM flood_dev.dev_<you>_bronze.ingest_heartbeat AS h, LATERAL variant_explode(h.payload:results) AS r
GROUP BY 1 ORDER BY 1;

-- API backfill progress for one source (swap the table name)
SELECT count(DISTINCT regexp_extract(source_file, 'date=([0-9-]+)', 1)) AS days_landed,
       min(regexp_extract(source_file, 'date=([0-9-]+)', 1)) AS first_day,
       max(regexp_extract(source_file, 'date=([0-9-]+)', 1)) AS last_day
FROM flood_dev.dev_<you>_bronze.pub_flood_alerts WHERE source_file LIKE '%/backfill/%';
```

Bronze only shows what the pipeline has streamed so far. A tail of the local log, `~/floodsense-api-days.log`, is the quickest live view.

### Things that bit us (keep in mind when touching the fetchers)

- **data.gov.sg:**
  - It returns 403 to Python's default `Python-urllib` User-Agent, so `client.USER_AGENT` is sent on every request.
  - A key it doesn't recognise (mistyped, expired) isn't refused: requests go through at the no-key limit, and the only sign is 429s below the key's limit. `check_key.py` measures the tier a key really gets. The key goes in the `x-api-key` header. Real-time limits are 6, 12 and 30 calls per 10 s (no key, dev, prod); dataset downloads 2, 4 and 10 (guide.data.gov.sg, "API Rate Limits").
  - The rate limit is one budget per client across all data.gov.sg hosts (the metadata API counts too). A 429 body says `code 24`, "try again in 10 seconds".
  - `poll-download` answers HTTP 201.
  - A day with no data is HTTP 404 with body `code 17`.
- **No `__file__` on serverless:** a `python_file` task runs through `exec()` in an IPython kernel, so `__file__` is undefined. The entry scripts locate `src/` through `inspect` instead.
- **Flood-alert labels:**
  - The flood-alerts API carries only flash-flood events (with lat/lon circles). The 90% drain-level warnings ("[Risk of Flash Floods]") exist only on Telegram.
  - Flood-alert history in the API really starts around October 2025. Earlier dates answer HTTP 200 with zero records.
  - PUB wipes its Telegram channel from time to time and keeps only a few posts: between Feb and Aug 2022, in mid-2024, between 9 Oct and 13 Nov 2024, and around 23 Mar 2025 (since then only single posts have gone). On the channel, everything before id 1111 (19 Mar 2025) is gone except 16 posts. The live capture keeps posts after PUB deletes them, so keep `flood-live` running.
  - The channel was `t.me/sgflood` until 14 Nov 2023, then `t.me/pubfloodalerts` (titled "Gov.sg-PUB Flood Alerts" until 3 Apr 2024). Post ids carried over, so key posts on `message_id` alone. An unrelated channel later took `sgflood`, so `telegram-wayback` reads `sgflood` captures only up to the rename, and drops any capture whose channel title lacks "Flood Alerts".
  - The web archive never captured 264 ids before they were wiped: 58 from Oct 2021 to Feb 2022, 128 from Feb to about Aug 2024 (766–894), 29 from 5–19 Oct 2024 (927–955), and 49 single posts or album parts. The `telegram-wayback` summary lists them.
  - Headlines drift, so the silver patterns need every version. The 90% warning: "RISING WATERS / POTENTIAL FLASH FLOODS" (Nov 2021), "🌊 Risk of Flash Floods" (Dec 2021), "⚠️ Risk of Flash Floods" (Feb 2023), "[Risk of Flash Floods]" (Nov 2023). A flood: "🚨 Flash flood at the following location(s)", then "[FLASH FLOOD OCCURRED]" (Nov 2024). Subsided: "🔽 Flash Flood at … has subsided", then "Flash flood subsided at …" (Nov 2023).
  - Silver: a post recovered from the web archive was seen at its `captured_at`, not its `fetched_at`, so order post versions by `captured_at` where it's set.
- **UHSLC:** the research-quality file is under `rqds/pacific/` (the `indian/` path is a 404 page). There is no Tanjong Pagar data after 2023, so silver fills 2024 onward with utide predictions.
- **Free Edition quota:**
  - Dev-mode deploys leave the `flood-live` and `flood-daily` schedules paused. Unpause them in the Jobs UI when you want the feed running, or set `presets: trigger_pause_status: UNPAUSED` on the dev target.
  - One pipeline update runs at a time.
  - Exceeding the daily quota stops compute for the rest of the day. Outside demo windows, consider `live_schedule: "0 0/15 * * * ?"`.
  - Measured cost: a 1.2 GB yearly CSV took about 3 min to land on serverless and about 20 s to stream into bronze. So the yearly CSVs are cheap. The 5-minute live cadence and the long API backfill are what need watching.

## Silver

Silver runs in the same `flood-etl` pipeline as bronze (`src/pipeline/transformations/silver/`) and publishes to the `silver` schema (`dev_<you>_silver` in dev). As in bronze, a table is defined only once the bronze folders it reads have landed. The Python that reads alerts (`labels/`) is plain Python shared with the fetchers and tests; the pipeline imports it from `src/` and ships it to the executors with its UDFs.

| Table | One row per | From | How it refreshes |
|---|---|---|---|
| `forecast_areas` | NEA forecast area: label point (47) | 2-hour forecast `area_metadata`, yearly CSVs | materialized view |
| `rain_stations` | rain gauge: location, name, first and last day | rainfall API `stations`, yearly CSVs | materialized view; the CSV half is private and rescans only when a CSV lands |
| `water_level_sensors` | PUB drain sensor, SVY21 → WGS84 | static sensor list, newest version | materialized view |
| `rainfall_5min` | gauge × 5-minute timestamp: mm, `published_at` (CSVs only) | yearly CSVs 2021–2024, API day backfill, live ticks | Auto CDC: the latest fetch wins, and an update merges only new pages |
| `forecasts_2h` | area × issue × amendment: category, NEA code, `rain_class` | yearly CSVs, API day backfill, live ticks | Auto CDC |
| `weather_stations` | weather station: location, name, variables, first and last day | the four weather APIs' `stations`, yearly CSVs | materialized view; the CSV half is private and rescans only when a CSV lands |
| `weather_1min` | station × variable × minute: value, `published_at` (CSVs only) | yearly CSVs 2021–2024, API day backfill, live ticks | Auto CDC; within a CSV the later publication wins |
| `lightning_strikes` | strike: time, place, type, `published_at` | lightning API (from 1 Feb 2025) | Auto CDC |
| `land_use` | Master Plan polygon: land use, area, GeoJSON, days in force | URA MP2019 and MP2025, newest versions | materialized view |
| `flood_prone_area` | year: PUB's flood-prone hectares | PUB's dataset, newest version | materialized view |
| `telegram_posts` | post: category, pattern, problem, locations and stamps | `bronze.pub_telegram_posts` (archive, wayback, live) | materialized view |
| `telegram_review` | post a person should look at, with what to do | `telegram_posts` | materialized view |
| `api_flood_alerts` | API alert: description read like a post, PUB's circle | `bronze.pub_flood_alerts` | materialized view |
| `location_gazetteer` | alert location: lat/lon, method, area | alerts, hand fixes, API circles, sensors, `bronze.onemap_search` | materialized view |
| `alerts` | alert location (Telegram or API), placed or not | the above | materialized view |
| `area_alerts` | **the labels** (`contract.AREA_ALERTS_COLUMNS`): placed alerts, each once | `alerts` | materialized view |
| `tide_gauge_hourly` | hour at Tanjong Pagar, 1984-2023: sea level (m above UHSLC's zero), null where UHSLC marks it missing | `bronze.uhslc_tanjong_pagar_hourly` | materialized view |
| `tide_extremes` | NEA high or low water: time (SGT, to the minute), height (m above Chart Datum), each month as NEA last published it | `bronze.nea_tide_timings` | materialized view |
| `alerts_unplaced` | alert no method could place (the quarantine) | `alerts` | materialized view |

### Reading PUB's posts

`labels/alert_text.py` has one pattern per format PUB has used, named after the format's marker (`warning_rising_waters`, `warning_wave`, `warning_caution`, `warning_bracket`, `flood_siren`, `flood_bracket`, `subsided_arrow`, `subsided_plain`, `heavy_rain`, `heavy_rain_eased`). A post matches one of them or it is `unmatched`; nothing is guessed. An unmatched post never becomes a label: it waits in `silver.telegram_review` with a `hint` (the category its words suggest), so a new format shows up the first time PUB uses it, and so does a post that isn't an alert. Posts are keyed on `message_id` alone, so a channel rename doesn't touch silver.

When `telegram_review` has a post:

- **An alert in a new format:** add a `Format` to `FORMATS`, and the post to `tests/fixtures/pub_telegram_texts.jsonl` with a test. Re-run the coverage report. The next pipeline update reclassifies every post.
- **Not an alert** (an announcement, an advisory): add it to `data/static/telegram_manual_labels.csv` as `not_relevant`, then redeploy. A manual label wins over the patterns.
- **`no stamp`:** usable as it is; its alerts are timed at the posting (`alerts.timed_by = 'posting'`). Ids 96 and 1268 are like this.
- **`method = 'jev'`:** Jev read a live post no pattern reads, and its reading is in use (next section). It stays in review until a pattern or a manual label takes over; treat it as an alert in a new format.

### When no pattern reads a post (Jev)

PUB changes its wording now and then (10 formats since 2021), and a post no pattern reads would otherwise wait for a person while the live alerts go quiet. So `flood-live`'s `fetch` (`ingest/fallback.py`) parses each new post as silver will, and for one the patterns can't read (unmatched, or a format matched but found no place):

1. **Asks Jev once** (`labels/jev.py`), per message id and text hash (`_state/telegram_jev.json` keeps the keys): one System One request with the category question and one place question per candidate phrase. A failed call is landed with its error and not retried.
2. **Lands the call** in `landing/telegram_jev_parses/live/`, so `bronze.telegram_jev_parses` holds every use of Jev: when, the post, why the patterns failed, the candidates, the answers, the token usage and latency. Test calls go under `test/` (`purpose = 'test'`), which silver ignores.
3. **Emails the developers** (`ingest/notify.py`; [One-time setup](#one-time-setup)).

`silver.telegram_posts` then reads Jev's newest answers for the post's id and text with `jev.read_row`, so a change to the cut-offs re-reads every past call without asking again. The reading is used, as `method = 'jev'` with `pattern = 'jev'`, only where no pattern reads the post, no manual label exists, and Jev gave the category at least 0.6; places need 0.5. Its alerts then flow through `silver.alerts` into `silver.area_alerts`, the app's replays and the scoreboard. `jev_category`, `jev_confidence` and `jev_asked_at` show what Jev said either way.

Training: a Jev-read alert is a label like any other (the user's decision, 23 Oct), as long as it stays recorded as Jev's. `read_by` carries that from `silver.telegram_posts.method` through `silver.alerts` into `silver.area_alerts` (contract column `read_by`: `pattern`, `manual` or `jev`), so `SELECT * FROM silver.area_alerts WHERE read_by = 'jev'` lists every label Jev made. Once a pattern reads the post, its alerts become `read_by = 'pattern'` on the next refresh. A post nothing reads (unmatched, or an alert whose place couldn't be read) is the opposite case: Jev failed on it, its endpoint missing included, so `gold.label_ticks` marks the windows it could carry missing with `missing_reason = 'post not read'` (contract `UNREAD_POST`), as it would a lost post, and no model trains on them. None in dev on 23 Oct.

Check the record of Jev's use:

```sql
SELECT payload:fetched_at::timestamp AS called_at, payload:purpose::string AS purpose, payload:message_id::bigint AS message_id,
       payload:status::string AS status, payload:usage AS usage, payload:latency_ms::int AS latency_ms, payload:error::string AS error
FROM flood_dev.dev_<you>_bronze.telegram_jev_parses ORDER BY called_at DESC;
```

### Placing alerts

`silver.location_gazetteer` places each location, keyed by `location_key` (lowercased, abbreviations spelt out, so `Jln Boon Lay` and `Jalan Boon Lay` are one place), by the first of:

1. `manual`: `data/static/location_gazetteer_manual.csv`;
2. `pub_api`: PUB's circle from the flood-alerts API, for that location and for the Telegram post of the same alert;
3. `pub_sensor`: the water-level sensor whose name matches at 0.8 or more (`locations.name_match`), unless another sensor more than 1 km away matches as well;
4. `onemap`: the first OneMap query (`locations.search_queries`, best guess first) whose top result is in Singapore and is named like the query (road, name or building at 0.8 or more). The name check matters: OneMap's top answer for "Kings Rd" is Kings Home on Joo Chiat Road, and for "Bukit Timah Rd" an expressway gantry.

Its area is the NEA forecast area with the nearest label point. `onemap_km` says how far OneMap's answer lies from the chosen point, as a cross-check. To place an alert in `silver.alerts_unplaced`, add its location to the manual CSV and redeploy.

```bash
# OneMap answers for alert locations it hasn't placed; ~10 s a search without a token, so run it on a laptop
uv run python src/ingestion/backfill.py onemap --landing $L --profile flood_dev
```

Run it again after new posts land. It asks each unplaced location only the queries it hasn't been asked, so new locations, and better `search_queries`, are searched without repeating old searches. A later task in `flood-daily` could do the same.

**Hand-check the flash-flood locations** (about 40; `SELECT * FROM ...silver.location_gazetteer WHERE array_contains(kinds, 'flash_flood')`). OneMap places a road at one point, so a junction on a long road can land in the next area: "Junction of Boon Lay Way and Corporation Rd" is placed on Boon Lay Way's point, in Jurong East, though the junction is nearer Jurong West's. Two spellings of one place can also land in two areas ("KPE (towards ECP) near Buangkok East Drive" in Paya Lebar by its sensor, "... slip road to Buangkok East Drive" in Sengkang by OneMap). Rows with `onemap_query_rank` of 2 or more were placed by one road of several, so check those first. A fix goes in the manual CSV.

### Weather, lightning and land use

- **Weather** (`weather_1min`): NEA's automatic weather stations report air temperature (`air_temp_c`), relative humidity (`rel_humidity_pct`), wind speed (`wind_speed_kn`, in knots) and wind direction (`wind_dir_deg`, where the wind blows from) every minute; both wind variables are NEA's 10-minute means. Readings stay at that resolution, one row per station, variable and minute, so the four variables merge independently: a wide row would need all four in one flow. As with rain, the CSVs give `published_at` and the API doesn't (one 2024 humidity reading was published 19 hours late), and gold decides what a tick may use. The variables, their bronze tables and the bounds silver keeps are in `utilities/silver_sources.py`.
- **Stations** (`weather_stations`): one place can change id between variables and years. Changi's wind is S199 in the API and its temperature S24, which had all four in the CSVs. NEA's points for a station differ by up to about 0.6 km between lists (S06 Paya Lebar, S117 Banyan Road), so join stations to areas by distance, not by id.
- **Lightning** (`lightning_strikes`): a record every 2 minutes lists the strikes of the 2 minutes before it, and NEA publishes it about 2 minutes after it closes, so a strike's `published_at` is 2–4 minutes after it. It reaches about 0.95–1.65 N, 103.3–104.4 E: storms over Johor and the Riau islands show before they reach Singapore. In the 50 minutes before the 13 Apr 2025 floods it listed 12,225 strikes, 140 of them cloud to ground.
- **Land use** (`land_use`): 113,212 polygons in MP2019 and 113,394 in MP2025, 33 land uses, about 790 km² each. `area_m2` is URA's own (`SHAPE.AREA`, measured in SVY21). A tick reads the plan in force on its day (`valid_from <= day < valid_to`). For shares per area, `h3_polyfillash3` takes the GeoJSON as it is.
- **The national map** (`map_polygons`, `map_lines`, 25 Oct): the layer is SLA's `FOLDERPATH` without `Layers/`. Polygons: `Coastal_Outlines` (105: the main island at 643 km², its islands, and the Johor coast, which gold leaves out by name), `Hydrographic` (897: 275 canals, 250 ponds, rivers and sungei, reservoirs, marshes, quarries), `Parks_NaturalReserve` (568), `Airport_Runway`, `Central_Business_District`. Lines: `Contour_250K` (7,874 contours, every 20 m from −40 to 160 m: 1,417 km of the 20 m line, 395 km of 40 m, 80 km of 60 m), `Major_Road` (7,348), `Expressway_Sliproad` (1,118), `Expressway` (729), `International_bdy` (1). A line that landed in parts keeps a row per part; `length_m` is the whole line's. The inset of Pedra Branca (its frame and labels) is left out; the rock itself is at its true place, 50 km east, and gold's box around the island leaves it out too.
- **Not in silver:** SLA's vertical control points carry no heights, and the radar frames are only archived.

### Checking it

```sql
SET TIME ZONE 'Asia/Singapore';  -- days and years in SGT; warehouses default to UTC

-- posts by category and how they were labelled; the review queue
SELECT category, method, count(*) FROM flood_dev.dev_<you>_silver.telegram_posts GROUP BY ALL ORDER BY 3 DESC;
SELECT * FROM flood_dev.dev_<you>_silver.telegram_review;

-- the share of alerts each method placed (the data-rigour number), and what's left to place by hand
SELECT coalesce(method, '(unplaced)') AS method, count(*) AS locations, sum(alerts) AS alerts
FROM flood_dev.dev_<you>_silver.location_gazetteer GROUP BY ALL ORDER BY alerts DESC;
SELECT * FROM flood_dev.dev_<you>_silver.alerts_unplaced ORDER BY alert_ts DESC;

-- the labels by kind and year
SELECT kind, year(alert_ts) AS year, count(*) AS alerts, count(DISTINCT to_date(alert_ts)) AS days
FROM flood_dev.dev_<you>_silver.area_alerts GROUP BY ALL ORDER BY 1, 2;

-- weather readings by variable, source and year, and the stations
SELECT variable, source, year(ts) AS year, count(*) AS readings, count(DISTINCT station_id) AS stations
FROM flood_dev.dev_<you>_silver.weather_1min GROUP BY ALL ORDER BY 1, 2, 3;
SELECT * FROM flood_dev.dev_<you>_silver.weather_stations ORDER BY station_id;

-- lightning strikes per day, and each plan's land use
SELECT to_date(ts) AS day, count(*) AS strikes, count_if(type = 'G') AS cloud_to_ground
FROM flood_dev.dev_<you>_silver.lightning_strikes GROUP BY ALL ORDER BY 1 DESC;
SELECT plan, land_use, count(*) AS polygons, round(sum(area_m2) / 1e6, 2) AS km2
FROM flood_dev.dev_<you>_silver.land_use GROUP BY ALL ORDER BY 1, 4 DESC;

-- expectation results of the latest update
SELECT origin.flow_name, e.name, e.passed_records, e.failed_records
FROM event_log('<flood-etl pipeline id>'),
     LATERAL explode(from_json(details:flow_progress.data_quality.expectations,
       'array<struct<name: string, dataset: string, passed_records: bigint, failed_records: bigint>>')) AS t(e)
WHERE event_type = 'flow_progress' ORDER BY timestamp DESC;
```

Timestamps are instants (`TIMESTAMP`); set `spark.sql.session.timeZone` to `Asia/Singapore` before reading them into pandas, as `contract.py` reads naive times as SGT.

### Things that bit us (silver)

- **Auto CDC compares the `sequence_by` types of all flows into a table, nullability included.** The API's `fetched_at` comes out of the payload (nullable) and the CSV's file time can't be null, so the API side falls back to the file time.
- **ANSI mode is on in serverless pipelines:** `map[key]` and `array[0]` throw on a missing key or an empty array, and a bad `CAST` fails the update. Use `try_element_at`, `get` and `try_cast`.
- **UDFs and `src/`:** the executors can't import the floodsense package, so `utilities/silver.py` registers its modules with cloudpickle by value. Only functions of those modules may be UDFs; a function defined in `utilities/` would be pickled by reference and fail on the executors. Those modules also import each other's names, not each other (`from .alert_text import Parsed`, not `from . import alert_text as T`): a module object is pickled whole, its `__loader__` and `__spec__` with it, and on Databricks those belong to dbruntime, so the executors fail with `No module named 'dbruntime'` (1 Nov: `read_jev` in silver.telegram_posts). `tests/test_silver_udfs.py` checks each UDF.
- **OneMap search** answers without a token, but with only its top result, an `error` asking for a token, and about one search per 10 s (faster ones get HTTP 429). With a free account and token it allows 250 a minute, and returns every result, which would also let a junction be placed where its two roads meet.
- **OneMap's top hit isn't always the place searched:** "Kings Rd" gives Kings Home on Joo Chiat Road, "Bukit Timah Rd" an expressway gantry. An answer counts only when its road, name or building is named like the query.
- **The API's text isn't the post's:** the same flood is "Boon Lay Avenue" in the API and "Boon Lay Ave (Between Jalan Boon Lay and Boon Lay Drive)" on Telegram, so an API alert is matched to a post by kind, time and a loose name match, not by text or by area alone (two floods 9 minutes apart in Bedok on 12 Dec 2025).
- **NEA publishes rain readings late:** see the reading-lag item under Gold.
- **A yearly CSV can hold a reading twice:** NEA revised 88 weather readings minutes after publishing them, and the file keeps both versions, with the same file time. So `weather_1min` sequences by publication as well, and the revision wins.
- **NEA lists some lightning strikes twice in one record** (232 of 12,225 on 13 Apr 2025, same millisecond, place and type), so a strike is keyed on all four.
- **NEA re-published its early-2025 forecasts on 8 May 2025:** every API forecast issued from 1 Jan to about 7 May 2025 carries that day as `update_timestamp`, which hid them from gold's point-in-time join until gold read them from their issue plus 10 minutes (21 Oct; the `forecasts_2h` item in the checklist). Those four months held 7 of training's 36 flash floods.
- **SLA's vertical control points have no heights:** 314 points with a number, a status and an update time, every z 0. Elevation needs a DEM, or the national map's contours, which are only 20 m apart.
- **The national map's contours are densified to a vertex every 30 cm or so** (median step 0.3 m, 25 Oct), so one contour can hold about 900,000 vertices, 40 MB as one JSON line, while a bronze VARIANT holds 16 MiB. They land in parts (`static.feature_parts`), and `bronze.static_national_map_line` has no null payload. The census also showed an `Expressway` layer apart from `Expressway_Sliproad`, which byte-range samples of the file had missed: count a layer's rows in silver before relying on samples.

## Gold

Gold is its own pipeline, `flood-gold` (`resources/gold.pipeline.yml`, code in `src/pipeline/transformations/gold/`), publishing to the `gold` schema. It reads silver by name, so run it after `flood-etl`: `databricks bundle run gold_pipeline -t dev`, or wait for `flood-daily`. Every table is a materialized view; an update whose inputs didn't change is skipped.

| Table | One row per | What it's for |
|---|---|---|
| `area_rain_stations` | area × gauge within 5 km | which gauges an area reads (493 pairs) |
| `area_forecasts` | area × tick with a valid 2-hour forecast | the forecast a live fetch at the tick would have had |
| `tick_weather` | tick | the island-wide weather (mean temperature, humidity, dew point and wind over the stations), and how many lightning records of the last 30 minutes exist |
| `area_lightning` | area × tick with a strike within 30 km in the last 30 minutes | strikes within 10 and 30 km |
| `area_features` | area × tick with a reading in reach in the last hour | the models' inputs (`contract.AREA_FEATURES_COLUMNS`): rain and the forecast (required), and the rain of the hours before, the rain around the area, the weather and lightning (optional) |
| `feature_ticks` | tick of `label_ticks` | whether the tick's features are recorded, and why not |
| `label_coverage` | interval between two consecutive recovered posts | where the label history is recorded or missing, and why |
| `label_ticks` | tick, 27 Oct 2021 to the last complete midnight | whether the tick's label window is recorded, and its split |
| `area_labels` | area × tick of `label_ticks` | the labels: `alert_status`, `alert_label`, `flood_label` |
| `model_rows` | area × tick whose label and feature windows are both recorded | **what models learn from** (`split = 'train'`) and are tested on (`'test'`) |
| `area_land_use` | area × Master Plan × land use | the share of the area in each land use, for the app and the dashboard |
| `area_terrain` | area | the ground from the national map (25 Oct): how far the hills, the sea and a canal or river are, how high the hills rise, the shares of trunk road and park; carried into `area_features` (see [The ground](#the-ground-from-the-national-map)) |
| `flood_prone_trend` | year | PUB's flood-prone hectares and the change from the year before, for the dashboard |
| `tide_model` | one row: the utide fit on Tanjong Pagar 2005–2023 | constituents, trend, residual, and the fit as JSON |
| `tide_datum` | NEA high or low water, beside the model's | the datum offset, and how far the model is from NEA |
| `tide_levels` | tick, 2021 through next year | the tide, imputed where the gauge is silent; `predicted_m` and `minutes_to_high_tide` go to `area_features` |

Two private tables sit behind `area_features`: `gauge_rain` (each gauge's 15/30/60-minute totals at each tick, and since 23 Oct the rain of its 2, 5 and 23 hours before the last one, read for the area's own gauges and for the ones around it) and `area_nearby_rain`. They, `tick_weather` and `area_features` itself are defined in the floodsense package's `features.py`, which `flood-live`'s score task shares, so a live prediction reads what the models learnt from.

### Rain features

`rain_{15,30,60}min_mm` at tick t is the largest 15/30/60-minute total among the gauges within 5 km of the area's label point, from the readings whose timestamps fall in the window ending 10 minutes before t and that NEA had published by t (`published_at`, where the CSVs give it). Each reading is credited to the 12 ticks whose windows hold it. Sums are exact decimals, so a rebuild gives the same values. There is a row only where some gauge in reach had a reading in the last hour; a missing row reads as no rain.

`rain_prior_{3,6,24}h_mm` (23 Oct, optional; in the `extended` feature set only) is the rain of the hours before the last one: with `rain_60min_mm` they make the 3-, 6- and 24-hour totals, split at the last hour so the burst at hand isn't counted twice. Per gauge it sums the gauge's own 60-minute totals at the ticks 1, 2, … hours before (each as published by its own tick, so never more than a live fetch had), through a window over the ticks that share the tick's second of the hour; it is null unless the gauge has every one of those hours, and the area takes the largest among its gauges. Checked on 23 Oct through Databricks Connect against hand sums on three gauges over 9–11 Nov 2025: 7,875 values, all equal. **It runs the other way from the usual flood intuition:** given the last hour, more rain before it means fewer alerts (`gold.model_rows`, 23 Oct: at 1–10 mm in the last hour, 2.5 alert rows per thousand after two dry hours against 0.26 after 30 mm or more; at 30 mm or more, 16.1 against 7.2), because PUB alerts as a storm starts, not hours into it, and Singapore's drains empty fast. So it has no monotone constraint, and the chance may fall with it.

`storm_age_min` and `rain_surplus_{10,25,50,100}mmh_mm` (24 Oct, optional; in neither feature set, read with `add=storm`, `add=surplus` or a column name) are the storm so far (contract.STORM_HISTORY). Both are in `features.py`, so live scoring computes them as gold does.

- **`storm_age_min`** counts the minutes since the storm over the area began. A storm begins at the first tick some gauge within 5 km has rain in its 15-minute window after 60 minutes without any (`STORM_GAP_MINUTES`: PUB's flash floods subside within the hour). It is null once 60 minutes pass dry, and capped at 12 hours, so live scoring's day of readings is enough. It is a window over the area's own ticks, each point in time (`features.storm_age`). The signal it carries: with 30 mm or more in the last hour, 27.7 alert rows per thousand in a storm's first 30 minutes against 5.4 after four hours; with 10–30 mm, 12.4 against 3.2 (`gold.model_rows`, 24 Oct). The rain windows end an hour back, so they can't see that.
- **`rain_surplus_<rate>mmh_mm`** is a leaky bucket per gauge, and the area takes the largest. Each reading fills the bucket at the first tick it counts, its *arrival*: 10 minutes after its timestamp, or later where NEA published it late. The bucket empties at the rate and never goes below 0: S(t) = max(0, S(t−1) + arrived(t) − rate/12). That recursion is computed in closed form with two running windows per gauge (Lindley: the arrivals so far less the drain so far, less its running minimum), in whole units of 1/1200 mm, so it is exact (`features.rain_surplus`). The rates bracket what the drains might take, since PUB publishes no per-drain capacity. 10 mm/h is a slow store that remembers the whole storm. 50 mm/h is about the hourly rain that news reports have overwhelming drains (49–56 mm). 100 mm/h is the rainfall rule's 25 mm in 15 minutes. Drains built to the post-2011 standard take up to 45% more than older ones. The models may read the age and the surplus either way: no monotone constraint.
- **Checked 24 Oct through Databricks Connect** against plain loops on two flood days (30 Mar 2023, CSV-era; 13 Apr 2025, API-era). The 6 wettest gauges' arrivals and buckets matched to 1e-13 mm, including 127 readings NEA published after the lag. The storm age matched exactly on 19,740 area-ticks a day. Gold's new expectations (`storm_age_in_range`, `surplus_nested`, `forecast_rain_group*_known`) passed on all 26.7M rows. In training, the storm age is known on 16% of rows but 77% of alert rows. The 10 mm/h surplus peaks at 601 mm, probably the Bedok/Tampines gauge flagged on 23 Oct.

`drain_flow_{10,30}min_mmh`, `drain_overflow_{10,30}min_mm` and `drain_rise_{10,30}min_mmh` (25 Oct, optional; in neither feature set, read with `add=drain` for the flows and overflows, `add=rise` for the rises) are the drains' response (contract.DRAIN_RESPONSE). The surplus's bucket drains at its rate whatever it holds, so it models the water that overflows a drain of that capacity, not the drain's level, which PUB's 90% warnings are about: at a drain's real capacity the bucket stays empty until the drain is already full, and at a low rate it holds a storm for hours after the drain has emptied. These model the level.

- **`drain_flow_<k>min_mmh`** is a nonlinear reservoir per gauge (SWMM's runoff model), and the area takes the largest. Each reading arrives as for the surplus. Its first 2 mm after a dry spell wet the ground's hollows, which dry at 4 mm a day. The rest is stored, and flows out as dS/dt = −a·S^(5/3), as an open channel's flow rises with its depth (Manning). So the store empties fast while full and slowly near empty, and a heavier storm peaks sooner and less damped, which no linear filter of the rain (the rain windows among them) can do. One parameter sets a: the response time k, the time constant at a steady 50 mm/h, 10 min for a small drain's catchment and 30 min for a canal's. The column is the store's mean outflow over the next tick from the water already in (the exact decay over the tick), in mm/h rounded to 0.1, so steady rain flows out as it comes in. It stands for the drain's level, since a level rises with its flow and the trees read only the order of a column. A store whose flow rounds to 0 without new rain has drained (after the heaviest storm, 6 h at 10 min and 19 h at 30 min), so live scoring, whose stores start empty a day back, computes what gold does.
- **`drain_overflow_<k>min_mm`** is the flow beyond 50 mm/h, still waiting: the surplus's bucket fed by the drain's inflow instead of the rain, in whole steps of flow.
- **`drain_rise_<k>min_mmh`** is the flow less its highest at the storm's earlier ticks (as far back as the age's 12-hour cap), 0 before the storm's first, null outside a storm (`features.storm_age`). PUB posts a warning when a drain first crosses 90%, so a level the storm has already passed is likely posted.
- **How it runs:** a nonlinear recursion has no closed form in windows, so `features.drain_response` runs `gauge_drains` in pandas, one gauge per applyInPandas group (about 600k ticks, a few MB), and joins back. The function carries its constants as a closure, so cloudpickle sends it by value (the executors can't import the package). Python UDFs don't run through Databricks Connect from the laptop (25 Oct: the sandbox fails to start, "exec format error", even for a plain `F.udf`), so check them in a one-off serverless job instead.
- **The alert chance may only rise with all three** (`gated.RISING`): more water in the drains, or a new high in the storm, never lowers it.
- **Checked 25 Oct.** Locally, the dry decay matches a Runge-Kutta integration of the equation, steady rain at 50 mm/h flows out at 49.8 mm/h (the hollows' evaporation), and 64% of a step is through in k minutes (`tests/test_features.py`). In a serverless job on environment 6 (flood-live's), the applyInPandas results equal `gauge_drains` run on the driver on two flood days (77,510 and 66,210 gauge-ticks, no nulls). Gold's expectations (`drains_known`, `rise_in_storm`) passed on all 26.7M rows; the rises recomputed from gold's rows by a plain loop matched on 1,942 values; and live and gold agree on all six columns over the 4 Dec 2025 storm (4,462 area-ticks). In training the rise is known on 16% of rows and 77% of alert rows, as the storm age is. **The signal:** at the same last-hour rain, ticks where the 10-minute flow sets a new high in the storm alert 5–8 times as often as ticks below an earlier high (under 10 mm: 4.11 against 0.54 alert rows per thousand; 10–30 mm: 14.4 against 2.4; 30 mm or more: 41.8 against 7.9).

### Forecast features

`forecast_rain_class` at tick t is the rain class (0 none to 5 heavy thundery; `utilities/forecast_categories.py`) of the newest 2-hour forecast NEA had published for the area by t (`updated_at` ≤ t), while its 2-hour valid period lasts; `forecast_age_min` is how old it was. NEA publishes all 47 areas at once, about every 25 minutes and a median 9 minutes after the nominal issue time.

`forecast_rain_group3` and `forecast_rain_group4` (24 Oct, optional; in neither feature set, read with `add=forecast_groups` or a column name) are the class in fewer groups (`contract.FORECAST_GROUPINGS`). Group3 is dry, rain (classes 1–3) and thundery (4–5). Group4 is dry or light (0–1), moderate or heavy (2–3), thundery (4) and heavy thundery (5). NEA's 24 wordings are already 6 ordered classes, and the trees split the class as one number under a rising constraint, so they can merge neighbours on their own. Only class 3 is thin in training: 15 alert rows on 4 area-days, against 221 and 229 area-days for classes 4 and 5. Alert rows per thousand by class: 0.03, 0.14, 0.84, 1.6, 4.0, 11.4 (`gold.model_rows`, 24 Oct). The groups keep the order, so the chance may only rise with them too. To swap one for the class: `gated:drop=forecast_rain_class:add=forecast_rain_group4`.

### Rain around the area, weather and lightning

These are the **optional** features (`contract.OPTIONAL_FEATURES`): null wherever their source has no data, which LightGBM reads as missing. Only the rain near the area and the forecast are **required** (`contract.REQUIRED_FEATURES`): `gold.feature_ticks` covers those, `gold.model_rows` keeps a row only where they are known, and the pair rates a row only where they are. So a source that starts late, or whose backfill hasn't landed, costs no row. Every value is what a live fetch at the tick could have had. Lightning is computed here too, but the models don't read it yet (`contract.HELD_BACK_FEATURES`, 16 Oct), so it isn't in `gold.model_rows`.

- **Rain around the area:** `rain_nearby_30min_mm` is the largest 30-minute total among the gauges 5–20 km from the area's label point, read as the rain columns (published by t, ending 10 minutes before it); 0 where none of them rained. `rain_upwind_30min_mm` is the same among those within 45° of the direction the island-wide wind blows from, and null when that wind is under 2 knots: then the stations disagree, and there is no upwind. During storms they often do (on 22 Nov 2024 upwind was null from 14:00 to the alerts), so the nearby rain carries the signal: of the 661 dry training rows an alert followed (no rain within 5 km in the hour), 76% had rain 5–20 km away, against 15% of quiet dry rows, and among dry rows the alert rate with 15 mm or more nearby is about 95 times that with none (16 Oct).
- **Weather:** `air_temp_c`, `rel_humidity_pct`, `dew_point_c` (23 Oct), `wind_u_kn` and `wind_v_kn` are means over NEA's weather stations, each station's newest reading in the 5 minutes ending 10 minutes before t that NEA had published by t (`gold.tick_weather`). Island-wide: 19 stations can't give 47 areas their own, and they come and go. Wind is averaged as a vector, from NEA's 10-minute means of where it blows from. The dew point (Magnus formula) is computed per station from its own temperature and humidity, then averaged: relative humidity is mostly the temperature again (correlation −0.986 over the 289 ticks of 10 Nov 2025, against 0.08 between the temperature and the dew point), so the `extended` feature set reads the dew point instead (`contract.EXTENDED_FEATURES`); the default `base` set keeps humidity, as the tree models did better on it (23 Oct). The means are of exact decimals, so a rebuild gives the same values.
- **Lightning:** `lightning_10km_30min` and `lightning_30km_30min` count the strikes within 10 and 30 km of the label point in the 30 minutes before t, each from the moment NEA published its 2-minute record. They are null before 1 Feb 2025, and wherever one of the 15 records of the window is missing (`gold.tick_weather.lightning_records`), since a missing record means a feed outage, not a quiet sky. Strikes are binned to a 0.01° grid first, so a distance is off by at most 0.8 km.

Checked on 16 Oct against hand recomputations from silver: the nearby rain on 6 random area-ticks of June 2023, and the temperature, humidity and wind on 5 random ticks of March 2022, all equal.

### The ground from the national map

`gold.area_terrain` (25 Oct, `terrain.py`) gives each area six numbers about its land, and `gold.area_features` carries them at every tick (`contract.MAP_FEATURES`, optional, in neither feature set; read with `add=map_line` for the first three, `add=map_polygon` for the rest). An area's land is its Voronoi cell, as in `area_land_use`: Singapore's coastal outlines filled with H3 cells of resolution 11, each placed in the area whose label point is nearest (`utilities/gold.nearest_area`, which `area_land_use` now shares).

- `high_ground_km`: the mean distance from the land to the nearest contour of 20 m or more on Singapore's land (so the Johor hills don't count). From 0.01 km in Bukit Panjang and 0.02 in Bukit Timah to 1.5 in Geylang and Marine Parade and 2.6 on Jurong Island.
- `relief_m`: the highest contour within 5 km of the label point, the reach of the area's gauges; 0 where none of 20 m or more is. 160 m around Bukit Timah Hill, 40 m in the east, 20 m in Tuas and on Jurong Island.
- `road_share`: the share of the land cells (about 50 m across) an expressway, slip road or major road crosses: 0.17 in Geylang, Kallang and Toa Payoh, 0.01 in the Central Water Catchment, 0 on the islands.
- `coast_km`: the mean distance to the sea (the land cells with a neighbour that isn't land; Tuas's and Semakau's enclosed basins count as sea). 6.8 km in the Central Water Catchment, 6.4 in Bishan, 0 to 0.1 on the southern and western islands.
- `waterway_km`: the mean distance to the nearest canal or river (hydrographic polygons named canal, river, sungei, creek, waterway or drain; 508 of them). 0.3 km in Kallang, Sengkang and Toa Payoh, 6 to 7 km on the southern and western islands.
- `green_share`: the share of the land in parks and nature reserves: 0.83 on Pulau Ubin, 0.63 in the Central Water Catchment, 0.32 in Bishan, under 0.01 in Geylang.

Distances are measured between resolution-9 cell centres (174 m edges), so each is good to about 0.2 km, and averaged as exact decimals; shares are ratios of counts. So a rebuild gives the same values. The whole table builds in about 40 seconds (prototype through Databricks Connect, 25 Oct). Three things limit what the ground can add. It is one value per area, and the area prior already gives each area its own alert rate, so it can only help where it changes what the weather does (a high tide near the sea, a downpour below the hills). The contours are 20 m apart, so they tell the hills from the lowland, not a low road from its neighbour. And the map is a 2025–26 snapshot read at every tick since 2021.

### Missing labels versus quiet ones

Telegram numbers a channel's posts without gaps, and a deleted post's id is never reused. So:

- two recovered posts with consecutive ids prove nothing was posted between them: that span is **recorded**, however long it is;
- an id gap means posts were never recovered (PUB deleted them before the web archive captured them). They could have gone out at any moment between their neighbours, so the **whole** span is **missing**, however few posts are lost. This is the prudent rule: the lost post between ids 1337 (2 Nov 2025) and 1339 (13 Nov 2025) blanks all 11 days, though it probably went out on one of them.

A tick's label reads alerts stamped in the next hour. Their posts can go out up to `POST_DELAY_MAX_MINUTES` (60; 35 seen) after the stamp, and a stamp can run up to 10 minutes ahead of its post. So a tick is `missing` if any of the posting time from 10 minutes before it to 2 hours after it falls in a missing span. It is also `missing` if an alert in the next hour hasn't been placed in an area (`silver.alerts_unplaced`), since it could be any area's. That reason shrinks as geocoding finishes, and hand fixes clear the rest.

`gold.area_labels.alert_status` then reads: `alert` (a placed alert came in the area within the hour, known even in a missing window), `no_alert` (none came, and the window was recorded), or `missing`. `alert_label` and `flood_label` are 1, 0 or null to match.

On 15 Oct: 54 id gaps (277 posts) blank 331 days of posting time. With the margins that is 334 days of ticks, nearly all in 2021–2024 (Feb–Oct 2024 is the big one), plus 1.3 days blocked by the 50 alerts not yet placed.

### Missing features versus dry ones

An `area_features` row exists wherever a gauge in reach reported in the last hour, zeros included: the yearly CSVs and the API carry every gauge's 0.0 mm. So a missing row means no data, not a dry tick. NEA's feed has outages (1–9 Jan 2023 has no reading at all, and there were 36 shorter gaps between Oct 2021 and May 2025), forecasts have their own, and the API backfill lands later days over hours. `gold.feature_ticks` marks a tick `missing` when the gauge network didn't report at every reading timestamp its rain windows read (`rain feed gap`), or no forecast was valid (`forecast gap`). Like id gaps, outages are island-wide. A single area whose gauges are all down while the network reports gets no row: the models say Low there, as they would live.

`scoreboard.score` takes these ticks as `blind=`. An event counts only if the model saw every tick that could have put a warning in force at it, from an hour before PUB's first alert to the event. Blind time isn't time at risk.

Why the forecast is a core feature (training ticks with rain features, 15 Oct): of 2,587 ticks an alert followed, 689 (27%) had no rain within 5 km in the hour the tick reads. That share grows with the time to the alert, from 6% within 15 minutes of it to 48% at 45–60 minutes: the early ticks, the ones that give lead time, are the dry ones. The forecast tells them apart: 52% of dry alert ticks had a forecast of rain (class 1 or more), against 7% of dry quiet ticks, and among dry ticks the alert rate under a thundery forecast (classes 4–5) is about 50 times that under a fair one.

On 15 Oct, of the training split's 1,180 recorded label days, 763 have features. 210 are blind for want of forecasts, mostly Jan–Jun 2025, and 207 for want of rain, mostly Jul–Dec 2025 (`flood-etl` hadn't streamed the backfill past June). The whole test is blind. Both fill in as the API backfill lands and `flood-etl` and `flood-gold` run.

On 16 Oct, with the rain and forecast backfill streamed through 14 Oct 2026: the test has features for 289.6 of its 295.4 recorded days (98%, past `train.py`'s 95% bar), and training for 970.5 of 1,180.3. Of training's 200 days blind for want of forecasts, 124 are Jan to 7 May 2025, whose forecasts NEA re-published on 8 May 2025 (the `forecasts_2h` item in the checklist): fixing their publication times would recover them, March 2025 included.

### Tides

The tide comes from one gauge, Tanjong Pagar (UHSLC station 699), and is the same for every area. UHSLC's research-quality record stops at the end of 2023, and NEA publishes only this month's and next month's high and low waters, so `gold.tide_model` fits utide (default constituents, linear trend, nodal corrections) on the 2005–2023 hours, a whole 18.6-year nodal cycle, and predicts every minute from 2021 to the end of next year. On 20 Oct the fit left 8.9 cm of surge (residual std) on a 71 cm tide, M2 0.80 m, S2 0.32 m, K1 0.31 m, O1 0.30 m; it gains 3.4 mm a year. Against NEA's Oct–Nov 2026 table, all 236 high and low waters matched, within 0.11 m (NEA prints 0.1 m) and 19 minutes. UHSLC's zero is 0.005 m below Chart Datum, which `gold.tide_datum` measures again on every update.

`gold.tide_levels.tide_m` is imputed tick by tick, and `source` says how:

- `observed`: the tick is on a valid UHSLC hour (2021–2023);
- `interpolated`: between two valid hours, the prediction plus the gauge's surge interpolated between them (more accurate than a straight line through the hourly heights, which cuts across the tide's curve);
- `predicted`: the prediction alone, in the gauge's gaps and on every tick from 2024 on.

**The models read `predicted_m`, not `tide_m`.** No gauge reading arrives live, and none exists after 2023, so a model trained on 2021–2023 surges would lean on an input it never gets when scoring, and live/backfill parity would break. The astronomical tide is known in advance, so `area_features.tide_height_m` and `minutes_to_high_tide` (to the next predicted high water) are honest at every tick. `tide_m` is there for analysis, and for a later surge feature if a live sea-level feed turns up.

First run in a workspace (both sources land in a few seconds, so a laptop is fine; `flood-live` keeps NEA's table fresh after that, and `flood-backfill`'s `uhslc` task does the same as the first line):

```bash
L=/Volumes/flood_dev/dev_<you>_bronze/landing
uv run python src/ingestion/backfill.py uhslc --landing $L --profile flood_dev
uv run python src/ingestion/fetch_live.py --landing $L --profile flood_dev --sources nea_tide_timings
databricks bundle deploy -t dev -p flood_dev
databricks bundle run etl -t dev -p flood_dev             # bronze and silver tide tables
databricks bundle run gold_pipeline -t dev -p flood_dev   # the model, the datum check, tide_levels, area_features
databricks bundle run train -t dev -p flood_dev           # the pair now reads the tide
```

```sql
SET TIME ZONE 'Asia/Singapore';
SELECT fit_start, fit_end, observations, residual_std_m, slice(constituents, 1, 4) FROM flood_dev.dev_<you>_gold.tide_model;
-- how far the model is from NEA's table (expect within 0.15 m and 30 min)
SELECT kind, count(*), max(abs(height_error_m)), max(abs(time_error_min)), any_value(datum_offset_m)
FROM flood_dev.dev_<you>_gold.tide_datum GROUP BY ALL;
-- how each year's tide was made
SELECT year(ts), source, count(*), round(stddev(surge_m), 3) AS surge_sd FROM flood_dev.dev_<you>_gold.tide_levels GROUP BY ALL ORDER BY 1, 2;
```

The fit and the predictions run in pandas on the driver (`daisi_team_master_corgis_floodsense/tides.py`): one fit of 166k hours (about 4 s) and 746k rows of predictions, made 10,000 minutes per utide call. Not as Python UDFs: on serverless (21 Oct) the fit as an applyInPandas ran out of worker memory, and the per-month predictions as an applyInPandas wrote 0 rows without an error, which left `tide_levels` empty and `model_rows` with no rows. Check `gold.tide_levels` is non-empty after a change here. A pair records its inputs, so one saved before the tide columns still loads; retrain (`flood-train`) after `flood-gold` to use them. The rater may only rise with `tide_height_m`, and a tide phrase in a reason ("high tide in 25 min") only ever follows a weather phrase.

What the tide adds: `flood-train` refits the chosen pair without each group in `--compare-without` (job parameter `compare_without`, default `tide`; groups in `contract.FEATURE_GROUPS`) and logs the comparison to the run (`without_tide_comparison.json`, `without_tide_*` metrics; `ml/compare.py`). Until the workspace is reachable, `explorations/tide_ablation_local.py` runs the same comparison on landed files with one island-wide area: on 21 Oct, over 2021–2024, the tide gave a small, consistent edge (ROC-AUC +0.0004; at a 5% warning budget 98.9% of 89 events flagged before PUB against 95.5%, McNemar p ≥ 0.125). Alerts were more common at low tide, so the `RISING` constraint on `tide_height_m` should go. Full write-up in `DEMO_DAY_NOTES.md`.

### The split

80% of the recorded label time trains and the last 20% tests, cut at an SGT midnight, with `split.EMBARGO_DAYS` (1) between them. Recorded time, not calendar time, so the months of missing labels in 2021–2024 count for neither side. `split.temporal_split` computed the cut on 15 Oct; it is pinned in `contract.py`, so later data never moves a tick between train and test.

| Split | Dates (SGT) | Recorded days | Share |
|---|---|---|---|
| train | 27 Oct 2021 – 18 Dec 2025 | 1,181.4 | 79.94% |
| embargo | 19 Dec 2025 | 1.0 | — |
| test | 20 Dec 2025 – 12 Oct 2026 | 295.4 | 19.99% |
| after_split | from 13 Oct 2026 | — | live scorekeeping |

Recorded days here count only the id gaps as missing; the cut ignores unplaced alerts, which geocoding changes. Placed alerts on 15 Oct: train 643 warnings on 162 days and 35 floods on 20 days; test 119 warnings on 35 days and 8 floods on 6 days (2026's census: 120 warnings on 35 days, 8 floods on 6 days). That is close to Round 1's "test on 2026".

To recompute the cut (only if recovered posts change):

```python
from daisi_team_master_corgis_floodsense.ml.split import temporal_split

# ticks: gold.label_ticks as pandas, ts in naive SGT; unplaced alerts are a geocoding gap, so they count as recorded here
recorded = (ticks["missing_reason"] != "posts not recovered").set_axis(ticks["ts"])
temporal_split(
    recorded[recorded.index < "2026-10-13"]
)  # Timestamp('2025-12-20 00:00:00')
```

### Training and scoring from gold

```sql
-- what models learn from: label window and feature window both recorded (flood-train reads this)
SELECT * FROM flood_dev.dev_<you>_gold.model_rows WHERE split = 'train';

-- the ticks to pass as scoreboard.score(..., missing=..., blind=...) and baseline.fit(...)
SELECT ts FROM flood_dev.dev_<you>_gold.label_ticks WHERE label_window = 'missing';
SELECT ts FROM flood_dev.dev_<you>_gold.feature_ticks WHERE feature_window = 'missing';
What the tide adds: `flood-train` refits the chosen pair without each group in `--compare-without` (job parameter `compare_without`, default `tide`; groups in `contract.FEATURE_GROUPS`) and logs the comparison to the run (`without_tide_comparison.json`, `without_tide_*` metrics; `ml/compare.py`). Until the workspace is reachable, `explorations/tide_ablation_local.py` runs the same comparison on landed files with one island-wide area: on 21 Oct, over 2021–2024, the tide gave a small, consistent edge (ROC-AUC +0.0004; at a 5% warning budget 98.9% of 89 events flagged before PUB against 95.5%, McNemar p ≥ 0.125). Alerts were more common at low tide, so the `RISING` constraint on `tide_height_m` should go. Full write-up in `DEMO_DAY_NOTES.md`.

```

Don't train on `area_labels` rows with `alert_label IS NOT NULL`. Inside a missing window that keeps the alerts that did come back (label 1) but none of the quiet ticks (null), so every lost post would raise the alert rate around it. `model_rows` keeps recorded windows only, alerts and quiet ticks alike, and its expectations fail the update if a missing one gets in.

`scoreboard.score` with `missing` judges an event only if every alert stamped from an hour before PUB's first alert was recorded, since an earlier alert could be lost: that is the label window of the tick an hour before, so a gap after the event doesn't count. It counts an unconfirmed warning as a false alarm only if its whole watch was recorded, and leaves missing time out of the time at risk. What it can't judge it counts in `alert_events_unjudged`, `warnings_unjudged` and the like. Rows for the rule can leave out `rain_60min_mm < 2.5`: every candidate threshold is at least 2.5 mm.

### Checking it

```sql
-- recorded and missing time by split
SELECT split, label_window, missing_reason, count(*) / 288.0 AS days
FROM flood_dev.dev_<you>_gold.label_ticks GROUP BY ALL ORDER BY 1, 2, 3;

-- the id gaps, longest first
SELECT from_ts, to_ts, after_message_id, before_message_id, missing_posts, hours
FROM flood_dev.dev_<you>_gold.label_coverage WHERE coverage = 'missing' ORDER BY hours DESC;

-- labels by status and split
SELECT split, alert_status, count(*) FROM flood_dev.dev_<you>_gold.area_labels GROUP BY ALL ORDER BY 1, 2;

-- how much of each split the models can use: label and feature windows, in days
SELECT l.split, l.label_window, f.feature_window, f.missing_reason, count(*) / 288.0 AS days
FROM flood_dev.dev_<you>_gold.label_ticks l JOIN flood_dev.dev_<you>_gold.feature_ticks f USING (ts)
GROUP BY ALL ORDER BY ALL;

-- how often each optional feature is known, by year (the backfill fills 2025-26 in)
SELECT year(ts) AS year, count(*) AS rows,
       avg((air_temp_c IS NOT NULL)::int) AS weather, avg((rain_upwind_30min_mm IS NOT NULL)::int) AS upwind,
       avg((lightning_10km_30min IS NOT NULL)::int) AS lightning, avg((rain_nearby_30min_mm > 0)::int) AS nearby_wet
FROM flood_dev.dev_<you>_gold.area_features GROUP BY ALL ORDER BY 1;
```

### Things that bit us (gold)

- **Bundle keys are unique across resource types:** the pipeline couldn't be called `gold`, since the schema is. It is `gold_pipeline`.
- **What blocks incremental refresh** (from the `planning_information` events): `max_by`/`min_by`, averages of floats, cross joins and Python UDFs. An update whose inputs didn't change is skipped either way.
- **Unplaced alerts weigh more than their count:** they fall in storms, so blanking their hour for every area removes the most informative quiet labels and events from scoring. The 50 left after OneMap cost 58 of 452 events in the training window, and the hours of 5 of training's 36 flash floods. Hand-fixed on 21 Oct (`data/static/location_gazetteer_manual.csv`, 29 locations): from the PUB sensor the alert names, OneMap with a cleaned query, or the gazetteer's own placing of the same place. Two sit on an area boundary (the KJE/PIE interchange, Bukit Batok or Jurong East; Sungei Tongkang at Ang Mo Kio Ave 5, Serangoon or Ang Mo Kio), noted in the file.
- **OneMap can place a generic road kilometres off:** the flash flood "Dunearn Rd (Sime Darby Centre)" (20 Jul 2023) was placed by OneMap's "Dunearn Rd" 2.8 km from Sime Darby Centre, in Novena rather than Bukit Timah. Corrected in the manual CSV; the other OneMap placings of bare road names deserve the same check.
- **Judge an event by the tick an hour before it, not by every tick in the hour:** a tick's label window reaches an hour ahead (two with posting delays), so checking all ticks before an event rejected events for gaps that came after them. Fixing that recovered 60 of 452 events.
- **A feature with gaps must be optional, not required:** `model_rows` drops every row with a null required feature, so lightning (from Feb 2025) as a required feature would have dropped every 2021–2024 row. New sources go in `OPTIONAL_FEATURES` unless a tick can't be judged without them.
- **A feature missing for a whole period can stand in for the period:** on 16 Oct the weather was known for 77% of training rows (2021–2024) and none of the test, as the 2025–26 weather backfill hadn't landed. Fitted then, the models could read "no weather" as "2025", the year with most of the floods, a pattern no live tick shows. Refit only once each optional input is known in both splits (`input_coverage.json`).
- **The surface wind is no guide to where a storm comes from:** in the hour before alerts the island-wide mean wind is under 2 knots 42% of the time (Nov–Dec 2024, against 28% otherwise), as the stations' winds scatter around a storm. Hence a null upwind there, and the direction-free nearby rain beside it.
- **Free Edition runs one serverless compute at a time:** with the SQL warehouse running (a query from the CLI starts it), a pipeline update fails at once with `RESOURCE_EXHAUSTED: You've hit the limit for serverless compute for free usage`. Stop the warehouse (`databricks warehouses stop <id>`) or let it idle out, then start the update.
- **Replaying live scoring over Jan–May 2025 finds no forecast:** `live.feature_rows` fetches forecasts by `updated_at`, and NEA re-published those months' forecasts on 8 May 2025, while gold counts them from their issue (`features.published_s`). So a live-vs-gold parity check there fails on every forecast column. Check parity outside that stretch (24 Oct: the 4 Dec 2025 storm). Scoring in real time is unaffected.


## Machine learning

`flood-train` (`resources/train.job.yml`, run by hand) refreshes `flood-gold`, then runs `ml/train.py` on serverless (environment 6: Python 3.12, plus `lightgbm`). It learns only from `gold.model_rows`, and reads the test split only when asked to.

```bash
databricks bundle run train -t dev -p flood_dev                              # v5's design (gated): fit, log, register
databricks bundle run train -t dev -p flood_dev --params '"models=gated,ordinal:alert=gated:flood=logistic"'   # and the hybrid
databricks bundle run train -t dev -p flood_dev --params score_test=true     # the single look at the test split
```

A run logs to the MLflow experiment `/Users/<you>/[dev <you>] floodsense`:

- the split (dates, folds), `split.summary`, and how much of each split the models can use (`coverage.json`, `usable_share_train`, `usable_share_test`);
- the rule, fitted on the development period: its parameters, its `rule_dev_*` scoreboard metrics and its threshold curve;
- the pair, once per gate recall tried (nested runs), then the one the scoreboard picked: `pair_*` (the gate's pass share and recall, rounds, cut-offs, Brier score), `oof_*` (out-of-fold scoreboard metrics with 95% CIs), the Medium curve, the reliability table, the inputs' importance and the event-by-event table;
- `input_coverage.json`: per model feature, the share of development rows where it is known, among all rows and among those an alert followed; an optional input known for under half of them is warned about in the log;
- the verdict: `beats_rule_lead`, `within_rule_false_alarms` and `beats_rule`. The pair's Medium cut-off may raise no more false alarms than the rule does, so the comparison is lead time at matched false alarms.

- in a sweep (24 Oct), every fit after the first is **paired with the first** on the same events and rows (`train.versus_first`, `compare.py`). Each nested run and `sweep.json` get `vs_first_*`:
  - the difference in rank (`rank_difference`: events flagged before PUB, a flood counting 10);
  - in PUB events and floods flagged, with the events only one fit flagged and an exact McNemar p;
  - in every row's log loss, which no cut-off chooses.

  Each comes with a 95% interval and a two-sided p from a bootstrap over whole days. Each nested run also logs `oof_events.json`. So put the reference first, and refit it with `seed=` (another sample of the quiet rows, another LightGBM seed) to see how far a refit alone moves the scores;
- `oof_floods.json` (`ml/floods.py`): every development flash flood one by one, out of fold: the lead time from the Medium warning in force at the flood (NaN for a miss, as the scoreboard counts it), whether it came before PUB's first alert, the tier and chance at the last tick before the flood, and their peak over the hour before it; with `oof_floods_*` counts over the judged floods.

Then it registers the pair in Unity Catalog (`flood_dev.dev_<you>_ml.dev_<you>_floodsense` in dev): every version as `@challenger`, and as `@champion` too when it beats the rule, unless `promote=false` leaves that to `flood-verify`.

### Verifying a registered pair before it goes live

`flood-verify` (`resources/verify.job.yml`, `ml/verify.py`) loads `@challenger` from Unity Catalog and fails if any of seven checks does: the pyfunc (flood-live's way) matches the saved pair; the model's logged code alone, in a fresh interpreter (the app's way), gives the same; the artifact's cut-offs and gate threshold are the ones its run logged; the area prior is finite (areas it never trained on are named: they score at the island's rate); the cut-offs are ordered; 5 mm more rain or a wetter forecast never lowers a chance; the outputs keep the contract. Then it scores the 3 hours before every flash flood in `silver.area_alerts` from `gold.area_features`, with the pair, the pair it would replace and the rule, and logs `floods.json` (one row per flood, with the out-of-fold columns from `oof_floods.json` beside) and counts over the floods none of them trained on (`verify_unseen_*`). With `promote=true` it moves `@champion` only if every check passes and the run beat the rule.

```bash
databricks bundle run train -t dev -p flood_dev --only train --params promote=false
databricks bundle run verify -t dev -p flood_dev                      # checks and the flood table, about 4 minutes
databricks bundle run verify -t dev -p flood_dev --params promote=true
```

### The ordinal model and the hybrid (23 Oct)

`ml/ordinal.py` has the details, and [ARCHITECTURE.md](ARCHITECTURE.md) the review, the evidence and the decision left open. Two rungs, each a binary model (Frank & Hall): the alert rung (`alert_label`, for Medium) and the flood rung (`flood_label`, for High) on every recorded row, each with its own credibility prior (the flood rung's, from flood days, gives most areas the island's rate). A tick is Medium when p_alert + 9 p_flood passes the cut-off, so flood risk can warn on its own; High is a cut-off on p_flood among warned ticks. A rung is LightGBM or a penalized logistic regression (`ml/logistic.py`); the alert rung can also be the whole gated pair (`alert=gated`): **the hybrid**, the gate kept as hard-negative mining for the rater but without its veto. Without the gated alert rung, quiet rows are sampled in two strata (stormy at 20%, dry at 2%) and weighted back up. Every fit tries the rainfall rule as a safety override (kept only if it flags more within the budget; none was, 23 Oct).

The default stays the gated pair on its base features (`gated`): out of fold it ranks first. The hybrid with a logistic flood rung (`ordinal:alert=gated:flood=logistic`) is the alternative for the flood targets: 5 of 13 development floods before PUB against 4, and 36 minutes of public warning before a flood against 27, for 11 of 322 alert events.

### The gated pair

v1 to v5 and the default (`ml/gated.py`; `gated` in a `models` spec, on the base features v5 reads):

0. **Area prior** (since 21 Oct, `ml/credibility.py`): each area's alert rate by Bühlmann–Straub credibility, one observation per area-day (claim 1 for a day with a PUB alert in the area, exposure the share of the day recorded), so an area's own history counts only as far as its exposure and the real spread between areas warrant. Both models start from it as an offset (LightGBM's `init_score`) instead of reading the area as a category, and each fold's prior comes from that fold's training days only. On 22 Oct (development period, recovered gold) it gave each area 92% of its own history (k = 99 recorded days, against 1,074 an area), so the 15 areas that never alerted sit at 1/12 of the island's rate (Lim Chu Kang 0.08) instead of next to zero, and the most alert-prone at 5 times it (Serangoon 5.0, Jurong West 4.1, Bedok 3.6). With flood days weighing 10 in its claims, the process variance rises and the credibility falls to 0.72, a floor of 0.3. `--area-effect category` fits the pair as before.
1. **Gate:** a LightGBM on every training row (quiet rows subsampled to 5% and weighted back up). Its out-of-fold scores set its threshold, the highest that still passes the target share of the ticks an alert followed (`gate_recall`).
2. **Rater:** a LightGBM on the rows the out-of-fold gate passed, its alert chance never falling as rain, rain around the area, lightning or the forecast class rises.
3. **Calibration:** centered isotonic regression on the rater's pooled out-of-fold scores. Screened-out rows get the screened-out alert rate.
4. **Cut-offs**, by the scoreboard on the pooled out-of-fold probabilities, as `baseline.fit` sets the rule's: Medium flags the most PUB events before PUB within the false-alarm budget, a flash flood flagged counting 10 events (`scoreboard.FLOOD_EVENT_VALUE`, `--flood-value`); High is the lowest higher cut-off whose High warnings are followed by a flood at least 1 time in 3. Every cut-off lies above the screened-out rows' probability, so Medium never warns the rows the gate screened out.
5. Both models are refitted on the whole development period, with the rounds the folds chose.

Floods weigh 10 in choosing, not in training: `--flood-weight` (a flood row's sample weight in both models, the gate's recall and the prior's claims) defaults to 1, as 10 cost 12 points of events flagged and caught no more floods (22 Oct, below).

Inputs: the SGT hour and the `area_features` columns (`contract.MODEL_FEATURES`), on top of the area prior (the area as a category before 21 Oct). A row is rated only where the required ones are known (rain near the area and the forecast); the optional ones (rain around the area and upwind, the island-wide weather) may be null, which LightGBM reads as missing. The alert chance may only rise with rain, rain around the area and the forecast class (`gated.RISING`). A new gold column needs adding to `contract.AREA_FEATURES_COLUMNS`, and to `REQUIRED_FEATURES` only if a tick can't be judged without it; the models pick it up unless it is in `HELD_BACK_FEATURES` (lightning, on 16 Oct). The rows a scorer passes may carry held-back columns; the pair ignores them.

### Results so far

**25 Oct (night): final_2, Moderate from 3% instead of 1%.** Asked for: Moderate at 3% and Elevated at 5%, instead of final_1's 1% and 5%. The user named it final_2, so final_1 stays on v13.

- **Registered as v14, with the alias `final_2`.** `flood-train --only train` ran `gated:add=storm+surplus+drain+rise:medium=0.03:high=0.05` at seed 0, override off, `compare_without=` and promote=false (job run 899267285577995; MLflow run `dbfc630e`, renamed `final_2`). It is v12's fit again (102 gate rounds, 70 rater rounds); only Moderate's cut-off moved.
- **At Moderate (every warning):** 44.6% of 323 events before PUB (95% CI 39.3–50.0%), 11.7 minutes of mean lead, 5 of 13 floods before PUB (7 had a warning out when they hit), 1.77 false alarms per area-month. A PUB alert followed 5.4% of warnings. Against final_1's Moderate from 1%, that is 111 fewer events and 5 fewer floods before PUB, for 3.5 fewer false alarms per area-month. Elevated flags the same 5 floods before PUB, so Moderate from 3% warns of no flood before PUB that Elevated misses; it warns of them earlier.
- **At Elevated:** as final_1 (29.4% of events, 5.8 minutes, 0.91 false alarms per area-month, 7.5% followed). So the verdict is final_1's too: more lead than the rule, but more false alarms (0.907 against 0.869), and `beats_rule` is 0.
- **`flood-verify` passed 7/7** (run `347d83b1`, `alias=final_2`, checks only, against final_1). On the 5 judged 2026 floods no model trained on, final_2 had a warning out at 4 of them but before PUB's first alert at only 2 (mean lead 25 minutes, misses counting 0). final_1: 4 before PUB (40 minutes); the rule: 3 (31 minutes).
- **Going live** (the user's choice, made before training on the condition that `flood-verify` passed). The app was stopped (the warehouse had stopped itself), `@champion` moved to 14 at 21:31 SGT (`final_1` stays on 13), and one `flood-live` run (21:31–21:42; the silver refresh 7 minutes) scored the 21:40 tick for all 47 areas as `final_2`, none warned. The app was deployed and started at 21:46 SGT (deployment `01f1d07a…`), and its track-record loader returns `final_2`, version 14, with Moderate at 3%. The app needed no change: its wording fits both, and it reads the cut-offs from the live model's run. To go back to final_1: alias `champion` to 13, one `flood-live` run, an app restart.

**25 Oct (evening): final_1, v12 with the tiers as fixed chances, shown as Moderate and Elevated.** Asked for: v12 with Low below 1%, Moderate from 1% and Elevated from 5%, and the model called final_1.

- **Registered as v13, with the alias `final_1`.** `flood-train --only train` ran `gated:add=storm+surplus+drain+rise:medium=0.01:high=0.05` at seed 0, override off, `compare_without=` and promote=false (job run 356818407850137; MLflow run `bb604b4d`, renamed `final_1`). Its chances are v12's: its tiers score exactly as v12's `tiers.json` does at 1% and 5%. It was `@challenger` too, and `@champion` from about 20:36 SGT (Going live, below) until final_2 replaced it at 21:31 (above).
- **At Moderate (every warning):** 78.9% of 323 events before PUB (95% CI 74.2–83.0%), 36.4 minutes of mean lead, 10 of 13 floods before PUB, 5.28 false alarms per area-month. A PUB alert followed 3.0% of warnings.
- **At Elevated:** 29.4% of events (24.7–34.6%), 5.8 minutes, 5 of 13 floods, 0.91 false alarms per area-month, 7.5% of warnings followed. The rule: 18.3%, 1.9 minutes, 0.87.
- **It misses the verdict on false alarms.** With fixed tiers the verdict reads Elevated (`verdict_tier` High) and asks for more lead at no more false alarms than the rule. The lead is longer, but Elevated raises 0.907 false alarms per area-month against the rule's 0.869, so `beats_rule` is 0. v11 passed at 0.77 on v10's chances. So `flood-verify --params promote=true` would have left `@champion` on v12. The user made final_1 live anyway, by moving the alias by hand.
- **`flood-verify` passed 7/7** (run `ba1b83cd`, `alias=final_1`, checks only, against v12). On the 5 judged 2026 floods no model trained on, final_1 warned of 4, all before PUB's first alert (v12: none; the rule: 5, 3 of them before PUB).
- **The names.** The data keeps Medium and High: `ml.predictions`, the scoreboard, `flood-verify`, the rule and the `oof_high_*` metrics. Only the app says Moderate and Elevated (`content.ts`, `TIER_LABEL`). A registered version now goes by the alias the user named it by (`contract.version_name`, copied into the app's `backend/model.py`). So once final_1 is live, its rows in `ml.predictions`, the track record and the replays say `final_1`. Its UC version comment says what final_1 is.
- **The app's copy for fixed tiers** (deployed 20:57 SGT). Elevated reads "PUB is more likely to issue a flash-flood alert here in the next hour" and takes the advice Medium had, as v12's Medium cut-off was 5.3%. Moderate keeps "PUB may issue…" with lighter heads-up advice. Nothing says a flash flood is likely. The track record says "About 8 in 10", counts false alarms in days, says how often warnings ended in an alert, names the cut-offs and compares Elevated with the rule. Its limits no longer say it misses most alerts. This build suits final_1 only: under v12 it would call a 5.3% Medium "Moderate" and give it heads-up advice.
- **Going live** (the user's choice, 25 Oct evening). `bundle deploy --select jobs.live` synced the code, so `ml.predictions` says `final_1`. The app was stopped (the warehouse already was), `@champion` moved to 13 at about 20:36 SGT (`champion`, `challenger` and `final_1` now all point at 13), and one `flood-live` run (20:36–20:48; fetch 1 minute, the silver refresh 8, scoring 3) scored the 20:45 tick for all 47 areas as `final_1`: none warned, its highest chance 0.8%. A model's first run scores only the newest tick (`live.ticks_to_score`: nothing is scored under that name yet), so v12's rows stop at 19:35 and the app's newest tick is final_1's alone. Then `bundle deploy --select apps.floodsense_app` and `bundle run floodsense_app` started the app (deployment `01f1d073…`, 20:57 SGT). The app's own track-record loader, run against the registry, returns `final_1`, version 13, with the fixed-tier figures. The job's 15-minute cap left little room after an 8-minute refresh; a refresh much longer than that would cost the scoring task.
  - **To go back to v12:** alias `champion` to 12, one `flood-live` run (v12 catches up from 19:35), and the app's wording back. Databricks Apps keeps each deployment's source, and the 19:40 one is at `/Workspace/Users/44b01660-66fe-4747-be6b-14ad05a237c9/src/01f1d068edd612c7883d3d1976529fdb` (`databricks workspace export-dir` it over `src/app`, rebuild, redeploy).

**25 Oct (afternoon): the drains' response, tested; it doesn't improve v10, and v10 stays.** Asked for: a model of the drains truer to their physics than the surplus's constant drain (B: the nonlinear reservoir's flows and overflows, `add=drain`), and the flow's rise over its highest earlier in the storm (C, `add=rise`); see [Rain features](#rain-features). Two `flood-train` sweeps (`--only train`, override off, register=false), because the first ran out of memory on its eighth fit and the second's first attempt on its first (free-tier memory is best-effort; job runs 312708116073991 and 424096641221674, MLflow parents `156150c5` and `f025620b`). v10's design at seed 0 opened both and reproduced v10 exactly in each (86 of 323 events, 4 of 13 floods, ROC AUC 0.9382), as did its seeds 1 and 2 the 25 Oct sweep's. The bar, set before the run, as for the ground: summed over seeds 0–2, at least as many PUB events and floods flagged before PUB as v10's design at the same seeds; ROC AUC higher at every seed; a positive mean seed-matched rank difference.

| spec | seed | events flagged before PUB, of 323 | floods, of 13 | rank score | ROC AUC | alert PR-AUC |
|---|---|---|---|---|---|---|
| `gated:add=storm+surplus` (v10's design) | 0 | 86 | 4 | 0.277 | 0.9382 | 0.0266 |
| | 1 | 90 | 5 | 0.307 | 0.9378 | 0.0255 |
| | 2 | 97 | 4 | 0.302 | 0.9382 | 0.0271 |
| v10 + `drain` (B) | 0 | 91 | 5 | 0.309 | 0.9380 | 0.0269 |
| | 1 | 79 | 3 | 0.241 | 0.9380 | 0.0253 |
| | 2 | 93 | 4 | 0.293 | 0.9382 | 0.0266 |
| v10 + `drain+rise` (B and C) | 0 | 90 | 5 | 0.307 | 0.9383 | 0.0313 |
| | 1 | 88 | 4 | 0.282 | 0.9383 | 0.0285 |
| | 2 | 87 | 5 | 0.300 | 0.9382 | 0.0289 |
| `gated:add=storm+drain+rise` (B and C for the surplus) | 0 | 89 | 5 | 0.305 | 0.9383 | 0.0297 |
| | 1 | 89 | 3 | 0.264 | 0.9378 | 0.0286 |
| | 2 | 85 | 3 | 0.255 | 0.9382 | 0.0282 |

- **None passes the bar.** In total, B flags 263 events against v10's 273 and 12 floods against 13; B and C, 265 and 14; B and C for the surplus, 263 and 11. Matched by seed, the rank moves +0.032, −0.066 and −0.009 for B (mean −0.014; seed 1 significantly worse, p 0.009: +4/−15 events, McNemar p 0.019), +0.030, −0.025 and −0.002 for B and C (mean +0.001, every p ≥ 0.27), and +0.027, −0.043 and −0.048 for B and C for the surplus (mean −0.021; seed 2 p 0.002).
- **B, the flows and overflows, adds nothing.** The trees hardly read them (in B and C, under 1% of the rater's gain at every seed), and B's ROC AUC fell at seed 0. On a drain's timescale a store is mostly a mix of the 15-, 30- and 60-minute windows the trees already split on.
- **C, the rise, ranks rows better, but not by enough to reach the warnings.** With it, ROC AUC rose at every seed (+0.00015, +0.00044, +0.00010), alert PR-AUC at every seed (0.0296 on average against 0.0264, +12%), flood PR-AUC at every seed (0.0088 against 0.0060), and log loss fell at every seed, matched by seed (−0.5, −1.2 and −0.4 × 10⁻⁵ nats a row, within what a reseed moves it, up to 1.0 × 10⁻⁵). The trees use it (7–8% of the rater's gain at every seed), mostly in place of the 50 mm/h surplus (its gain at seed 0 13,849 → 7,196) and the storm age (5,948 → 3,003). At looser budgets it flags a little more at 2 false alarms per area-month (45.5% against 43.8% of events, mean over seeds) but not at 1 (29.7% against 29.7%) or 4 (70.2% against 70.3%). The same pattern as the storm so far on 24 Oct: a consistent ranking gain that the rule-matched cut-off, at under one false alarm per area-month, doesn't turn into warnings.
- **The signal is real, but the trees already had most of it.** In gold.model_rows the rise separates alert rates 5–8 fold at the same last-hour rain (Rain features, above); within v10's inputs, the 15-minute window, the storm age and the surplus at 50 mm/h already say most of it (rain intensifying early in a storm).
- **Left in place:** the six columns stay in gold, optional and in neither feature set (live scoring computes them too, and agrees with gold). Nothing was registered from the sweeps.
- **Then the user chose B and C anyway:** registered as v12 (`flood-train --only train`, seed 0, override off, `compare_without=`), and it reproduced the sweep's fit exactly (90 of 323 events, 5 of 13 floods, 0.83 false alarms per area-month, 5.4 minutes of mean lead). `flood-verify` passed 7/7 and moved `@champion` from v10 to v12 (25 Oct, 19:24 SGT). On the 5 judged unseen 2026 floods it warned of none (v10: none, the rule: 5). A warned row's reason can now name the rise ("rain here is the heaviest it has been in this storm", `gated._phrase`, once even when both rises raised the score); the flows and overflows make no reason. To go back: alias `champion` to 10, one `flood-live` run, an app restart.

**25 Oct: the ground from the national map, tested; it doesn't improve v10, and v10 stays.** Asked for: SLA's vertical control, the national map's polygons and its lines, taken from bronze to training. The vertical control can't be a feature: its 314 benchmarks carry no heights. The two map layers became six columns per area (`gold.area_terrain`; see [The ground](#the-ground-from-the-national-map)): `map_line` (high_ground_km, relief_m, road_share) and `map_polygon` (coast_km, waterway_km, green_share). One `flood-train` sweep (MLflow parent `4ff1365a`, job run 865745369609683, `--only train`, override off, register=false) ran on the same 13,916,772 training rows as v10's own fit, and v10's design at seed 0 reproduced v10 exactly: 86 of 323 events, 4 of 13 floods, ROC AUC 0.9382. The bar, set before the run: summed over seeds 0–2, at least as many PUB events and floods flagged before PUB as v10's design at the same seeds; ROC AUC higher at every seed; a positive mean seed-matched rank difference.

| spec | seed | events flagged before PUB, of 323 | floods, of 13 | rank score | ROC AUC | alert PR-AUC |
|---|---|---|---|---|---|---|
| `gated:add=storm+surplus` (v10's design) | 0 | 86 | 4 | 0.277 | 0.9382 | 0.0266 |
| | 1 | 90 | 5 | 0.307 | 0.9378 | 0.0255 |
| | 2 | 97 | 4 | 0.302 | 0.9382 | 0.0271 |
| v10 + `map_line+map_polygon` | 0 | 86 | 5 | 0.298 | 0.9381 | 0.0265 |
| | 1 | 83 | 3 | 0.250 | 0.9379 | 0.0270 |
| | 2 | 89 | 3 | 0.264 | 0.9383 | 0.0267 |
| v10 + `map_line` | 0 | 93 | 5 | 0.314 | 0.9381 | 0.0281 |
| v10 + `map_polygon` | 0 | 93 | 4 | 0.293 | 0.9384 | 0.0274 |

- **Both groups fail the bar on every count.** Matched by seed, the rank moves +0.021, −0.057 and −0.039 (mean −0.025). In total 258 events against 273, and 11 floods against 13. ROC AUC fell at seed 0 (−0.00007) and rose by at most 0.0002 at the others.
- **Each group alone looks better at seed 0, by no more than a refit moves it.** Lines +0.036 in rank (95% CI −0.016 to +0.098, p 0.29), 7 more events and 1 more flood; polygons +0.016 (−0.005 to +0.038, p 0.16), 7 more events. v10's own design moved +0.030 from seed 0 to seed 1 and +0.025 to seed 2, the second with p 0.025: a reseed alone looks significant here. The lines' ROC AUC fell at seed 0, so they can't pass the bar.
- **Why so little:** the ground is one value per area, and the area prior already gives each area its own rate, so the ground can only add where it changes what the weather does. Contours 20 m apart can't tell a low road from its neighbours.

**24 Oct: the storm so far and coarser forecasts, tested; none significantly improves the model, and v5's design stays.** One `flood-train` sweep (MLflow parent `bbb132df`, job run 925704565646949, `--only train`, override off, register=false). It ran out of fold with 323 judged events and 13 judged floods, at the rule's budget (0.86). Every fit is paired with the first, v5's design at seed 0 (`vs_first_*`). Each pairing gives the rank difference with a 95% day-bootstrap interval and p, with a Holm correction over the 8 feature variants.

| spec | events flagged before PUB | floods, of 13 | rank score | rank vs first (95% CI) | p | alert PR-AUC |
|---|---|---|---|---|---|---|
| `gated` (v5's design), seed 0 | 28.8% | 5 | 0.314 | – | – | 0.0251 |
| same, `seed=1` | 29.7% | 4 | 0.300 | −0.014 (−0.071, +0.028) | 0.76 | 0.0259 |
| same, `seed=2` | 26.6% | 4 | 0.277 | −0.036 (−0.088, +0.000) | 0.07 | 0.0259 |
| `add=storm` (storm age) | 27.9% | 4 | 0.286 | −0.027 (−0.079, +0.011) | 0.26 | 0.0262 |
| surplus at 10 mm/h | 26.9% | 3 | 0.259 | −0.055 (−0.116, +0.009) | 0.11 | 0.0255 |
| surplus at 25 mm/h | 29.1% | 4 | 0.295 | −0.018 (−0.078, +0.024) | 0.61 | 0.0262 |
| surplus at 50 mm/h | 28.8% | 3 | 0.273 | −0.041 (−0.103, +0.020) | 0.26 | 0.0260 |
| surplus at 100 mm/h | 26.6% | 4 | 0.277 | −0.036 (−0.091, +0.003) | 0.10 | 0.0276 |
| forecast in 3 groups | 25.4% | 4 | 0.268 | −0.045 (−0.095, −0.004) | 0.04 (Holm 0.30) | 0.0265 |
| forecast in 4 groups | 27.2% | 3 | 0.261 | −0.052 (−0.113, +0.011) | 0.14 | 0.0257 |
| all three, seed 0 | 28.5% | 4 | 0.291 | −0.023 (−0.080, +0.019) | 0.43 | 0.0262 |
| all three, `seed=1` | 30.7% | 5 | 0.327 | +0.014 (−0.010, +0.036) | 0.32 | 0.0265 |
| all three, `seed=2` | 25.1% | 4 | 0.266 | −0.048 (−0.103, −0.007) | 0.02 | 0.0265 |

"All three" is `gated:add=storm+surplus+forecast_rain_group4:drop=forecast_rain_class`.

- **Refitting moves the scores as much as any feature does.** v5's design at three seeds ranks 0.277–0.314 (26.6–29.7% of events). Matched seed for seed, all three minus v5's design is −0.023, +0.027 and −0.011 (mean −0.002). The day bootstrap treats a fit as fixed, so one fit's p understates the noise: seed 2 of v5's own design came out at p = 0.07 against seed 0. Before reading a rank difference under about 0.04 (3 points of events) as real, refit with `seed=`. That applies to the earlier single-fit sweeps too, such as 23 Oct's base 30.1% against extended 25.5%.
- **Storm age:** the trees use it (6% of the rater's gain, 7th of 16 inputs), but it adds nothing at the Medium cut-off. It mostly restates what the rain windows, the hour and the forecast already say.
- **Surplus:** no drain rate helps; 25 mm/h comes closest. 100 mm/h ranks rows best (alert PR-AUC 0.0276, against 0.0251–0.0259 for v5's design over seeds) but flags fewer events within the budget.
- **Forecast groups:** both cost events. The 3 groups merge thundery with heavy thundery, which alert at 4.0 and 11.4 rows per thousand. That is the one nominally significant result, and it runs the wrong way (p = 0.04, Holm 0.30).
- **Without a cut-off:** every storm or surplus variant has a slightly lower log loss than v5's design at seed 0, by 0.4–1.8 × 10⁻⁵ nats a row (all p ≥ 0.17). The two seed refits differ by +4.9 and −4.0 × 10⁻⁶. That may be a trace of signal, but nowhere near significant.
- **Floods:** no variant flags a flood that v5's design (seed 0) misses (`vs_first_flood_flagged_only_with` is 0 throughout). All three at seed 1 matches its 5.
- **Diagnostics** (`diag_*`, `ml/diagnostics.py`; a re-run of the same sweep, MLflow parent `76199abb`, reproduced every fit exactly). They are much the same for every fit:
  - Rows at Medium: about 6,000 of 13.9M rows warned (0.04–0.05%). False-positive rate 0.037–0.047%, false-negative rate 90–92%, precision 5.6–6.3%.
  - Warnings: 1,380–1,580 Medium warnings, 92% of them false alarms. PUB events missed: 224–242 of 323. Floods: 3–5 flagged before PUB, 1–2 warned only after PUB's first alert, 6–9 missed.
  - **High never fires:** no cut-off reaches 1 in 3 flood precision, so no fit has a High warning. The exception is surplus at 100 mm/h: 3 High warnings and 1 hit, from 8 rows that calibration put at chances up to 1.0 (kurtosis 10,608 against about 600).
  - Chances: skewness about 19 over every row (median 3.1 × 10⁻⁵, since 95% of rows are screened out by the gate) and about 5 over the rows the gate passes. ROC AUC is 0.937–0.938 for every fit, and the alerts expected match those that came (0.98–1.01).
  - Medium curve: v5's design flags 29–34% of events at 1 false alarm per area-month, 45–47% at 2 and 67–70% at 4 over its three seeds. No variant beats that at more than one budget.
- **Storm age and surplus without the forecast swap** (`gated:add=storm+surplus`, 24 Oct evening, MLflow parent `286efd72`; the forecast groups had cost events on their own). Three seeds against v5's design at the same seeds, with the bar for registering set before the run: at least as many events and floods in total, no seed significantly worse, and ROC AUC up at every seed.
  - ROC AUC rose at every seed (+0.0007 to +0.0011), and floods held (13 against 13).
  - Events fell slightly: 273 against 275, by seed −7, −6 and +11 (seed 2: +17/−6 events, McNemar p 0.035). Rank differences were −0.036, +0.007 and +0.025 (p 0.16, 0.96, 0.46). It never issued a High warning.
  - So it narrowly fails the events bar and was not registered. The ranking gain is consistent but doesn't reach the warnings.
  - **Then the user chose it anyway:** registered as v10 (`flood-train --only train`, seed 0, override off), and it reproduced the sweep's fit exactly (86 events, 4 floods, 0.84 false alarms; beats the rule). `flood-verify` passed 7/7 and moved `@champion` from v5 to v10 (24 Oct, 21:12 SGT). On the 5 judged unseen 2026 floods it warned of none (v5: 1, the rule: 5). `flood-live` scored with it, and the app was redeployed (21:29 SGT).
- **Left in place:** the columns stay in gold for now. They are optional and in neither feature set, so v5 and live scoring are unchanged. Live and gold agree on them over the 4 Dec 2025 storm (4,462 area-ticks, storm ages up to 480 minutes). The harness keeps `seed=` and `vs_first_*`. Nothing from these sweeps was registered at the time (v10 below came after).

**24 Oct (night): the tiers reviewed for v10.** `tiers.json` (`diagnostics.tier_table`, logged for every gated fit since then; run `2dd8d0b7`, the same fit as v10) scores other tier definitions on v10's out-of-fold chances, over the development period (323 events, 13 floods).

- **Medium** is the most events flagged within the rule's false-alarm rate: p ≥ 4.9%, 26.6% of events, 4 floods, 5 minutes of mean lead, 0.84 false alarms per area-month, 7% of warnings followed by a PUB alert. As a fixed chance instead:

  | Medium from | events before PUB | floods, of 13 | mean lead (min) | false alarms / area-month | warnings an alert followed | rule at that false-alarm rate |
  |---|---|---|---|---|---|---|
  | 0.5% | 87.6% | 11 | 58 | 7.9 | 2% | 70.3%, 21 min |
  | 1% | 81.1% | 11 | 40 | 5.4 | 3% | 54.5%, 12 min |
  | 2% | 59.8% | 6 | 21 | 3.0 | 4% | 36.8%, 6 min |
  | 3% | 45.8% | 6 | 13 | 1.8 | 5% | 25.7%, 4 min |
  | 5% | 26.0% | 4 | 5 | 0.77 | 8% | 11.5%, 2 min |

- **High can't fire as defined:** at no cut-off do 1 in 3 High warnings end in a flood. The best flood precision on p_alert is 1 in 20 (p ≥ 10.4%, 20 warnings, catching no flood before PUB). At p ≥ 5%, 7 of 1,418 warnings end in a flood (0.5%) and 110 in any PUB alert (8%). High where the rule's rain (25 mm in 15 minutes) falls on a Medium tick: 7 of 432 (1.6%). The app's "A flash flood is likely here in the next hour" has nothing behind it at any cut-off.
- The 2026 floods are descriptive only, as the test is spent: v10's peak chance in the hour before each judged one was 2.2, 3.5, 4.0, 3.6 and 0.5%. That is none at 4.9% and four at 1%, though not necessarily before PUB's first alert.
- **Decided by the user: the tiers by chance.** Low is below 1%, Medium from 1% and High from 5% (`gated:...:medium=0.01:high=0.05`, `gated.Settings.medium_chance` and `high_chance`). An 8% fourth tier was weighed and dropped: barely riskier than 5% (12% against 8% of warnings followed by an alert), in force before PUB for 6% of events, and in the sparse top of the calibration.
  - **The verdict reads High** when the tiers are fixed chances (`verdict_tier`, `oof_high_*`), since Medium then warns about six times as often as the rule. Sweeps still choose Medium within the rule's false-alarm rate by default, so fits compare as before.
  - **v11** (MLflow `77b3b4c7`): v10's design with the new tiers. At Medium: 81.1% of events and 11 of 13 floods before PUB, 39.7 minutes of mean lead, 5.37 false alarms per area-month, 3.0% of warnings followed by a PUB alert. At High: 26.0% of events, 5.0 minutes, 0.77 false alarms per area-month (the rule: 18.3%, 1.9 minutes, 0.87), 7.8% followed. It beats the rule at High. `flood-verify` passed 7/7 and moved `@champion` to v11 (24 Oct, 22:32 SGT). On the 5 judged 2026 floods it warned of 4, all before PUB (v10: 0; the rule: 5).
  - **Then back to v10** (the user's choice, about half an hour later): `@champion` was set to v10 again, v11 stays `@challenger`, `flood-live` caught v10 up (21:25–23:05), and the app's v11 wording and track-record changes below were taken out again. The app was redeployed 23:10 SGT with the same build v10 ran with (identical bundle hashes). The `medium=`/`high=` options, the verdict at High and `oof_high_*` stay in the harness for a model with fixed tiers.
  - **The app, as it was with v11:** High read "PUB is more likely to issue a flash-flood alert here in the next hour" ("PUB alert more likely"). High carries the advice Medium had before, since that is the same chance, and Medium gets lighter heads-up advice. The track record shows "about 8 in 10", false alarms in days, the rule compared with High at about the same false-alarm rate, and how often warnings came true. Redeployed 22:46 SGT after one `flood-live` run.

**23 Oct (late evening): labels placed by sensor, PUB's alerts nearby as inputs, a recency prior; v5's design still holds.** From the hotspots and near-miss review (2026 test, descriptive: the old hotspots weren't quieter once two reworded sensors were re-placed; v5 warns within 3–5 km more than its own area).

- **Placement.** Hand rows for CWS046 and CWS163 (Things that bit us), and the gazetteer's `sensor_near_onemap`: a warning's location goes to its best-named sensor (name score ≥ 0.5) when OneMap's answer lies within 1 km of it. 91 locations (304 alerts) moved to their sensor's point, 0 unplaced; 43 of 857 alerts changed area (17 in the 2026 test). Posts listing several roads with ";" stay one alert: they are one sensor's roads ("avoid this location"), as `test_labels_alert_text` pins.
- **`pub_alerts`** (contract.PUB_ALERTS): PUB alerts posted for other areas within 5 km in 30 and 60 min and within 10 km in 60 min, counted from POST_LAG_MINUTES after posting, null unless gold.label_ticks an hour earlier is recorded. One definition in features.py (`nearby_alerts`, `recorded_posts`); live counts from silver.area_alerts, which gained lat/lon. Live and gold agree on every model feature on two 2026 storms (2,325 area-ticks). In neither feature set; a spec reads them with `add=pub_alerts`.
- **Recency prior:** gated `half_life=` (credibility.recency; a fold's prior weighs both ways from the middle of its block, the final one back from its last day).
- **The sweep** (MLflow parent `479df9c1`, `--only train`, register=false), out of fold on the corrected labels, 323 judged events and 13 floods, at the rule's budget:

| spec | events flagged before PUB | floods, of 13 | false alarms / area-month | rank score | alert PR-AUC | flood PR-AUC |
|---|---|---|---|---|---|---|
| `gated` (v5's design) | 28.8% | 5 | 0.84 | 0.314 | 0.0251 | 0.0049 |
| `gated:add=pub_alerts` | 31.0% | 4 | 0.86 | 0.309 | 0.0264 | 0.0059 |
| `gated:half_life=730` | 26.3% | 3 | 0.81 | 0.255 | 0.0253 | 0.0063 |
| `gated:add=pub_alerts:half_life=730` | 29.4% | 5 | 0.83 | 0.318 | 0.0258 | 0.0047 |

  The nearby alerts rank alerts a little better and flag 2 more points of events but one fewer flood; the recency prior alone costs 2.5 points and 2 floods. Together they tie v5's design (0.318 against 0.314, within noise), so nothing was registered and v5 stays `@champion`. As a rule on its own (5 km, 15 min), the same signal flagged 21.4% of development events at 0.65 false alarms, so its value may be as an override rather than an input: untested.
- **App:** each area's panel shows the strongest warning within 5 km beside its own tier (`add_nearby`, `/api/now`); deployed 23 Oct 21:55 SGT.
- **PUB's works line up with the quiet sites** (PUB Drainage Improvement Projects, updated 9 Oct 2025): Sennett Estate (CWS220 Puay Hee Ave) completed Oct 2023, Enterprise Road (WWS113) Dec 2024; ongoing: road raising at Lorong Gambir (CWS435, 3Q 2027), Lorong 1 & 2 Toa Payoh (CWS433, 2Q 2027), Jalan Boon Lay (WWS114, 2Q 2027), Eng Kong Place (WWS418, 1Q 2027), Yarwood Avenue (by WWS045, 2Q 2027).

**23 Oct (afternoon): the review's recommendations, tested; v5's design still ranks first, the hybrid is better on floods.** Three `flood-train` sweeps (MLflow parents `f568c845`, `4ee3b986`, `3f303a06`) over the same folds and scoreboard, at the rule's false-alarm budget (0.869), 322 judged events and 13 judged floods. [ARCHITECTURE.md](ARCHITECTURE.md) has the full table; the headline:

| out of fold | events flagged before PUB | floods before PUB, of 13 | public warning before a flood | rank score |
|---|---|---|---|---|
| v5's design (`gated`, base features) | 30.1% (25.4–35.3%) | 4 | 27.1 min | 0.303 |
| hybrid: gated alert rung + logistic flood rung | 26.7% (22.2–31.8%) | 5 | 36.2 min | 0.298 |
| ordinal, both rungs logistic, no gate (extended) | 26.1% | 5 | 34.3 min | 0.294 |
| ordinal, both rungs LightGBM, no gate (extended) | 19.9% | 3 | 31.6 min | 0.207 |
| rainfall rule | 18.0% | 5 | 27.6 min | – |

- Without the gate the trees lose 8–10 points: the gate is hard-negative mining. Penalized logistic rungs don't need it.
- A LightGBM flood rung (179 flood rows) adds nothing; a logistic one adds a flood and about 9 minutes of warning before floods.
- The extended features (dew point for humidity, prior rain) cost the trees 3–5 points; the prior rain is worth +0.6 points to the logistic model. Given the last hour, prior rain *lowers* the alert rate (storm onset).
- Floods weighing 3 in training: −2.5 points, no more floods. The rule as override: no threshold flags more within the budget.
- The first sweep ran the serverless driver out of memory (every fit copied the 13.9M development rows); fixed by sharing one sorted copy (`gated.development_rows`), float32 in the logistic learner and freeing the design before the cut-off search. Serverless retried the failed task from the start; that retry was cancelled.

**23 Oct: v5 verified and `@champion`, and better than v4 everywhere, but the rule still warns more flash floods.** `flood-train` (run `45fecb69`, `--only train`, gate recall 0.9) reproduced the 22 Oct credibility fit exactly (30.1% of 322 events flagged before PUB, the same cut-offs: the fit is deterministic) and registered it as v5; `flood-verify` passed all seven checks (run `b7941a07` failed one first: Pulau Tekong, whose one gauge reported only after training, has no prior, which is right; the check now names such areas instead of failing) and moved `@champion` from v4. Flash floods one by one (`floods.json`):

| flash floods | v5 (pair) | v4 (old pair) | rainfall rule |
|---|---|---|---|
| 13 judged development floods, a warning in force when it hit | 6, 4 before PUB, 29 min ahead on average (out of fold) | its run judged 1 and missed it; its design refitted on the recovered data flagged 1 of the 13 before PUB | 9, 5 before PUB, 24 min (its threshold fitted on them) |
| 5 judged 2026 floods (never trained on; the test was spent on 19 Oct) | 1: Tanglin 22 Sep, 26 min ahead, before PUB | 0 | 5, 3 before PUB, 31 min |
| peak chance in the hour before the 2026 floods | 0.9–4.7% (Medium from 4.5%) | 0.5–1.4% | (no chance) |

- The pair ranks storms better than the rule (30.1% of all PUB events flagged before PUB against 18.0%, at 0.84 false alarms per area-month against 0.87), but flash floods are the rule's case: the heaviest rain at the spot, which the pair's chance at the flood barely passes. Most missed floods peaked at 2–4%, about 100 times the base rate, just under the Medium cut-off the rule's false-alarm rate fixes. `medium_curve.json`: at 0.020, 9 of 13 development floods before PUB and 57% of events at 2.7 false alarms per area-month (5 of the 7 2026 floods warned); at 0.014, 10 of 13 and 76% at 4.4. The budget is the product decision that decides flood recall.
- Lim Chu Kang (30 Sep 2026, 85 mm in the hour) stayed at 0.9%: its 1,074 recorded days without an alert give it 1/12 of the island's rate at 92% credibility. Credibility treats that as evidence of a low-risk area; a cell like this floods anywhere, so an area effect that weakens with heavy rain (an interaction, not a constant offset) is the next thing to try.
- 18 of the 31 development floods still have no out-of-fold prediction at all: their hour is in a missing label window (the single-post id gaps, a decision still open).

**22 Oct: the pair was too conservative mostly because the floods never reached it.** Three `flood-train` sweeps (MLflow parents `2cd46ac8`, `1c7a035b`, `9d2e37f5`; `sweep.json` in each), all out of fold over the development period, at the rule's false-alarm budget unless said:

1. Today's gold (run A): the old model and the three changes were all within noise of each other, and the old model came out on top (19.8% of 227 judged events flagged before PUB). Only 1 development flood could be judged, and it had no rows, so nothing about floods could show.
2. After recovering the data (run B: the 50 unplaced alerts placed, the Jan–May 2025 forecasts read from their issue time): training rows 12.3 to 13.9 million, alert rows 2,358 to 4,102, flood rows 16 to 179, judged development events 227 to 322, judged floods 1 to 13. Then:

| gate recall 0.9, 322 events and 13 floods judged | events flagged before PUB | lead | false alarms / area-month | floods flagged | at 2 / 4 false alarms |
|---|---|---|---|---|---|
| area as a category, no flood weight (the old pair) | 26.1% | 6.3 min | 0.80 | 1 | 47.5% / 64.6% |
| **credibility prior, no flood weight (now the default)** | **30.1%** (95% CI 25.4–35.4%) | 5.7 min | 0.84 | **4** (13.5 min lead) | 45.3% / 70.5% |
| category, flood rows ×10 | 22.7% | 5.4 min | 0.80 | 3 | 42.6% / 61.5% |
| credibility, flood rows ×10 | 18.0% | 4.5 min | 0.78 | 4 | 39.8% / 59.6% |
| rainfall rule (`rain_15min_mm` ≥ 25) | 18.0% (14.2–22.6%) | 1.9 min | 0.87 | 5 (16.6 min lead) | |

3. The gate (run C, the credibility pair): recalls 0.85–0.99 all flag 26.7–30.1% of events, within each other's intervals, and 0.9 is best. A stricter gate (0.8, 3% of rows passed) screens out flood ticks: 2 floods flagged. Passing everything an alert ever looked like (recall 1.0, 84% of rows) is worst, at 23.3%, so the gate earns its place by letting the rater learn storms.

- Weighting flood rows ×10 in training made the pair worse and caught no extra flood once the prior was in: about 20 training floods are too few to reweight. It also dragged the gate open, since the gate's recall must then pass nearly every flood row: the category ×10 gate passed 68% of rows. So floods now count 10 times in choosing fits and cut-offs (`FLOOD_EVENT_VALUE`) but weigh 1 in training.
- The false-alarm budget is still what holds the pair back most: the same pair flags 45% of events at 2 false alarms per area-month and 70% (29 min lead) at 4.
- Still unjudged: 148 of 470 development events and 18 of 31 floods. Every id gap next to a flood is a single missing post, minutes wide, and in the intact channel (ids from 1111) 13 of its 15 single-post gaps come right after an alert post, 3 of them after a flash flood: posts PUB deleted (a pin notice or a correction, most likely), not alerts the scrape lost. Reading those as recorded would bring about 10 more training floods to the pair (see [Missing labels versus quiet ones](#missing-labels-versus-quiet-ones)); it relaxes the 15 Oct rule, so it waits on a decision.
- The 2026 test was spent on 19 Oct and helped find these problems (the misses at Lim Chu Kang and Tanglin), so it is no fresh measure of this pair; the after-split days from 13 Oct 2026 are.

| 15 Oct, out of fold over the development period (usable time: Nov 2021 – Dec 2024) | gated pair, gate recall 0.9 | rainfall rule, `rain_15min_mm` ≥ 25 |
|---|---|---|
| PUB alert events flagged before PUB's first alert (192 judged) | 20.8% (95% CI 15.7–27.1%) | 9.4% (6.0–14.3%) |
| mean lead before PUB's first alert | 5.1 min | 0.9 min |
| false alarms per area-month | 0.87 | 0.91 |
| High | never: one flood event could be judged | never |

- 260 of the development period's 452 alert events can't be judged: id gaps, and 2025, whose features hadn't landed.
- The trade-off is steep. The same pair flags 37% of events with 15 minutes' lead at 2.0 false alarms per area-month, and 48% with 26 minutes at 3.1 (`medium_curve.json`). The budget is a product decision: Round 1 only promises to report false alarms.
- Gate recalls 0.85, 0.9 and 0.95 flagged 19.8%, 20.8% and 18.8%, well within each other's intervals. The gate passes 7%, 12% and 21% of area-ticks.
- The inputs that matter most (gain): the area, the forecast class, the hour, then 15-minute rain.

This isn't the model to test yet. 2025 is a third of the training time and holds most of its floods. Once the API backfill has landed rain and forecasts through 12 Oct 2026:

1. `databricks bundle run etl -t dev -p flood_dev` (silver);
2. `databricks bundle run train -t dev -p flood_dev` (gold, then a refit). Check that `usable_share_train` is near 1 and `usable_share_test` at least 0.95, that every optional input is known in both splits (`input_coverage.json` for training; the weather needs its 2025–26 backfill), and that the pair still beats the rule;
3. only then, once: `databricks bundle run train -t dev -p flood_dev --params score_test=true`.

### Reading a run's artifacts

The workspace's public DBFS root is disabled, so `databricks fs cat` can't read MLflow artifacts. MLflow's client can, with `mlflow-skinny` and `databricks-sdk` installed (the project environment has neither):

```python
import mlflow

mlflow.set_tracking_uri("databricks://flood_dev")
mlflow.artifacts.download_artifacts(
    run_id="<run id>", artifact_path="medium_curve.json", dst_path="."
)
```

### Things that bit us (ML)

- **pandas versions:** until 15 Oct the project ran pandas 3.0, while serverless environment 6 has 2.3. On pandas 2.3, `merge_asof` refuses `by` keys of different dtypes, and `train.py` loads the area as a category (to fit ten million rows) while alerts carry it as text. The scoreboard now compares areas as plain objects. The project environment now pins environment 6's versions (see [One-time setup](#one-time-setup)), so `uv run pytest` tests at the job's versions.
- **The model's signature** declares every feature a double. MLflow refuses int64 input (int32 and floats are fine), so cast before calling the pyfunc.
- **Bundle keys again:** the experiment couldn't share the key `floodsense` with the registered model; it is `training`.
- **Plain isotonic calibration** gave thousands of storm ticks the same probability, so no cut-off could fall between them and the Medium curve jumped from 45% flagged to 22%. Centered isotonic fixed it.
- **An outage looks like a dry spell:** `area_features` has no row where no gauge reported, which reads as no rain unless `feature_ticks` says otherwise (see [Gold](#missing-features-versus-dry-ones)).
- **The flash floods barely reached training:** on 21 Oct only 16 of the training split's 390 flood rows (the hour before 5 of its 36 floods) were in `gold.model_rows`. The rest fell in missing label windows (single-post id gaps right at the floods; 5 floods sat under an unplaced warning) or in the Jan–May 2025 forecast gap, so only 1 development flood event could be judged, High could never be set, and on the 2026 test the pair flagged none of the 5 judged floods while the rule flagged 3. Weighting floods (`flood_weight`) does nothing for a flood that isn't in the rows: check `pair_flood_rows` and `oof_flood_events` in a run before reading its flood metrics.
- **The area as a category reads an area's few warnings as no risk:** PUB's warnings come from its water-level sensors, so the category taught the models where the sensors are. On 30 Sep 2026 the pair rated Lim Chu Kang 0.2% with 85 mm of rain in the hour under a heavy-thundery forecast, eight minutes before a flash flood (the Medium cut-off was 2.3%). Hence the credibility prior (`credibility.py`); note that the category concentrates warnings where PUB alerts most, which the false-alarm count rewards (21 Oct, old gold: 19.8% of events flagged at the rule's budget, against 18.5% with the prior).
- **LightGBM doesn't boost from the average when it is given an `init_score`** (lightgbm 4.7, checked 21 Oct): the trees start at 0, so a model trained on offsets has to add its own base log-odds (`GatedPair.gate_base`, `rater_base`).
- **`lgb.cv` can't give a row a different `init_score` in each fold**, which an in-fold prior needs, so `gated._out_of_fold` grows the folds' boosters in step and stops them together as `lgb.cv` does; for the pair without a prior it reproduces `lgb.cv`'s fits exactly (19.82% flagged, 6.59 min lead, 0.789 false alarms on 21 Oct, as before).
- **PUB rewording a sensor's text can move its alerts to another area:** OneMap places each wording on its own, and a sensor near an area boundary lands either side. CWS046 (Upp Paya Lebar Rd/Rochdale Rd, nearest Hougang) became "Service Rd off Upper Paya Lebar Road (near Lim Teck Boo Road)" on 12 Dec 2025, which OneMap put in Geylang; CWS163 (Balestier Rd / Thomson Rd, nearest Toa Payoh) became "Thomson Rd (Balestier Rd to Novena Rise)" in 2024, put in Kallang. With labels as placed, the eight busiest training areas looked 19% quiet in 2026 (rain-adjusted); with these two re-placed they were on expectation. Fixed 23 Oct by hand rows and the gazetteer's `sensor_near_onemap` rule (silver alerts.py).
- **`databricks bundle run --params` splits on commas** when it gets more than one `key=value`, so pass one `--params` per key when a value holds commas (`--params models=gated,gated:extended --params compare_without=antecedent,tide`). A value that itself holds `=` (model options, `ordinal:flood=logistic`) is split even then (23 Oct: "ordinal:legacy must be formatted as key=value"); quote the whole pair for the CLI's CSV reader: `--params '"models=ordinal,ordinal:flood=logistic"'`.

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

GLO-30 is free with registration, 30 m, a surface model (includes buildings and trees), heights relative to EGM2008, and requires the Copernicus attribution notice. Use area-level summaries (min, mean, share of low-lying cells) rather than point values.

## List of Flood Prone Areas

The list (as at 2 Apr 2025) is 36 road/estate names with no geometry, so geocode with OneMap and assign each to an area. PUB defines these as low-lying areas with a history of flooding. The data.gov.sg file is only the yearly total (27 ha in 2022 down to 23.3 ha in 2025); an earlier version of the same dataset carried 2010–2021.

## PUB Water Level Sensors Locations

Seems to be down as of 11 October, 2026, 3:24pm. Iirc it was up when designing the model for Stage 1. Update: the data.gov.sg download API (`poll-download` for `d_31333fa5cf0834f012d840365b336610`) returned a working .xlsx link at 6:00pm SGT on 11 Oct 2026, so it may only be the web page that's down. It's a current snapshot, so it can't show when each sensor was installed (see [Flash Floods & Water Level Sensor Alerts](#flash-floods--water-level-sensor-alerts)).