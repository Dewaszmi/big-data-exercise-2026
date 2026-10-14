# Milestone 1: Data Profiling & Quality Audit
## Modern Data Engineering Lifecycle on BTS Airline Flight Delays
**Course:** Big Data Engineering 2026  
**Authors:** Student Project Team  
**Dataset:** US Bureau of Transportation Statistics (BTS) On-Time Statistics & Delay Causes (2003–2025)

---

## Slide 1: Project Scope (What We Are Building)
* **Goal:** Design and deploy an end-to-end Data Engineering Lifecycle pipeline on real-world US civil aviation telemetry data (~400,000 monthly operational records across 23 years, 425 airports, and all major US carriers).
* **Full Lifecycle Architecture:**
  * **Milestone 1 (Current):** Schema discovery with DuckDB, zero-copy in-process SQL audit, anomaly classification, and data contract formulation.
  * **Milestone 2 (Physical Optimization):** Multithreaded cleaning via Polars, conversion from raw CSV to Apache Parquet with Snappy compression, hive-style partitioning (`year` / `carrier`), and I/O partition pruning benchmarks.
  * **Milestone 3 (ML Serving):** Feature store engineering, historical delay rate indexing, and predictive ML model (Scikit-Learn) for delay classification and flight disruption estimation.

---

## Slide 2: Business Objectives & Value Proposition
* **The Business Problem:** Flight delays cost airlines and consumers over $30B annually in the US alone. Cascading delay propagation across hubs (e.g. ATL, ORD, DFW) causes crew timeouts and equipment misplacements.
* **Why This Pipeline Is Valuable:**
  * **Airlines & Airport Authorities:** Early anomaly detection in turnaround delays enables dynamic gate reallocation and contingency scheduling.
  * **Travel Tech & Passengers:** Transparent, data-driven delay risk scoring helps travelers make informed booking and connection decisions.
  * **Data Engineering Efficiency:** Transitioning from unoptimized raw files to partitioned Parquet reduces query latency by >90% and compute cost significantly.

---

## Slide 3: Data Schema, Source & Legal Licensing
* **Data Source:** US Bureau of Transportation Statistics (BTS) / Research and Innovative Technology Administration (RITA).
* **Dataset Scale:** 398,233 aggregate monthly records spanning from June 2003 to January 2025 (~51 MB uncompressed CSV).
* **Licensing:** US Federal Government Public Domain (17 U.S.C. § 105) & Kaggle Open Database License. Permitted for research, educational, and non-commercial analysis.
* **Core Schema Entity:**
  * **Identifiers:** `year` (INT), `month` (INT), `carrier` (VARCHAR, IATA/DOT code), `airport` (VARCHAR, 3-letter IATA code).
  * **Operational Volumes:** `arr_flights` (Total arrivals), `arr_del15` (Arrivals delayed $\ge 15$ min), `arr_cancelled`, `arr_diverted`.
  * **Delay Duration:** `arr_delay` (Total minutes delayed).
  * **Attributed Causes:** `carrier_delay`, `weather_delay`, `nas_delay`, `security_delay`, `late_aircraft_delay`.
  * **Proportional Incident Counts:** `carrier_ct`, `weather_ct`, `nas_ct`, `security_ct`, `late_aircraft_ct`.

---

## Slide 4: Architectural Deep-Dive: Why DuckDB?
* **In-Process Analytical Engine (OLAP):**
  * Unlike PostgreSQL or MySQL, DuckDB has no client-server network latency or socket serialization overhead.
  * Operates directly inside the application process memory.
* **Vectorized Query Engine:**
  * Uses columnar vector processing (SIMD) to execute filters and aggregations at hardware speeds.
  * Direct streaming from CSV/Parquet without full materialization in RAM (zero-copy query execution).
* **Memory Footprint:** Evaluates 400k rows in <0.2 seconds while using <50 MB of resident memory.

---

## Slide 5: Completeness & Quality Profiling (Hard SQL Statistics)
* **Total Records Analyzed:** **398,233 rows**
* **Completeness Profile:**
  * Identifiers (`year`, `month`, `carrier`, `airport`): **100% complete (0 NULLs)**.
  * Flight counts & delay statistics: **657 NULL values (0.16%)**.
  * Detailed delay indicator (`arr_del15`): **950 NULL values (0.24%)**.
* **Boundary Statistics (Min / Max / Mean):**
  * `year`: 2003 to 2025.
  * `arr_flights` per carrier/airport/month: Min = 1, Max = 21,977, Mean = 313.
  * `arr_del15` (flights delayed $\ge 15$ min): Min = 0, Max = 6,377, Mean = 60.
  * `arr_delay` (cumulative minutes): Min = 0, Max = 648,300 mins (~450 days of delay in 1 month at peak hub!), Mean = 3,708 mins.

---

## Slide 6: Technical Anomalies & Domain Interpretation
1. **Fractional Counts (`carrier_ct`, `weather_ct`):**
   * *Observation:* 327,994 rows (82.3%) contain fractional values (e.g. 1.45, 0.5) instead of integers.
   * *Domain Insight:* When a delay is co-caused by multiple factors (e.g. 50% weather, 50% late equipment turnaround), BTS algorithms proportionally apportion fraction weights. This is an intentional regulatory methodology, not a data corruption.
2. **Component Sum Mismatches:**
   * *Observation:* 9 rows have discrepancy $|arr\_delay - \sum(causes)| > 1$ minute (e.g. difference of 1,345 mins).
   * *Technical Insight:* Caused by retroactive carrier dispute adjustments or manual overrides in specific regional carrier filings.
3. **100% Cancellation Events:**
   * *Observation:* 270 carrier-airport months had 100% flights cancelled (`arr_cancelled == arr_flights`).
   * *Real-world Explanation:* Extreme weather disruptions (Polar Vortex, Hurricane Katrina) and the catastrophic April 2020 COVID-19 aviation freeze.

---

## Slide 7: Limitations: What the Data Does NOT Tell Us
* **Aggregate vs Individual Flights:** Data is aggregated monthly per carrier and airport; individual flight trajectories, tail numbers, and exact timestamps are not captured.
* **Micro-Weather Telemetry:** Records indicate weather delay duration but lack raw meteorological variables (wind speed, visibility, precipitation).
* **Passenger Impact & Compensation:** Does not reveal actual passenger load factors, stranded traveler counts, or hotel rebooking expenses.
* **Hidden Delays (Gate vs Tarmac):** Does not differentiate between taxi-out ground hold vs airborne holding patterns.

---

## Slide 8: Risks, Governance & Ethical Considerations
* **Fairness & Carrier Bias:** Small regional carriers operating exclusively in blizzard-prone mountainous regions (e.g. Aspen, Jackson Hole) naturally exhibit higher delay percentages than carriers operating in sunny Southern hubs. Raw metrics must not be used to penalize carriers without route-difficulty normalization.
* **Privacy & Compliance:** High privacy compliance: contains purely institutional and operational statistics; zero PII (Personally Identifiable Information) regarding passengers or flight crews.
* **Model Misuse Risk:** If deployed in insurance underwriting or dynamic ticket pricing, unadjusted delay models could unfairly inflate fares on regional connectivity routes.

---

## Slide 9: Project Challenges, Surprises & Lessons Learned
* **Surprise 1:** Discovering fractional numbers in count columns (`carrier_ct = 0.33`), requiring careful reading of BTS methodology documentation.
* **Surprise 2:** The speed of DuckDB: executing multi-column aggregations and filter clauses over 400k rows completed in under 150ms without creating any database tables.
* **Challenge:** Distinguishing true corrupted records (e.g. 9 component mismatch rows) from legitimate statistical edge cases (e.g. COVID-19 months with zero completed flights).
* **Next Steps:** Proceeding to **Milestone 2**: Polars data cleansing pipeline, Parquet Snappy conversion, and physical partitioning benchmarks.
