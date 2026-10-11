# Comprehensive Dataset Documentation & Provenance

**Course**: BAMAT207 - Probability and Statistics  
**Project**: Weather Prediction using Conditional Probability & Bayesian Intelligence  
**Evaluator**: Dr. Pallavi Mishra  

---

## 1. Inventory of Datasets Stored in `/data`

| Dataset File Name | Spatial Location & Coordinates | Temporal Span | Total Records | Observations per Day | Source / Provenance |
|---|---|---|---|---|---|
| **`weather_data.csv`** (Primary) | Pune/Western Ghats, India (18.52°N, 73.85°E) | Jan 1, 2015 – Dec 31, 2024 | 3,652 Days | Daily Aggregates | Open-Meteo European Weather Archive |
| **`weather_pune_10yr.csv`** | Pune Meteorological Station, India | Jan 1, 2015 – Dec 31, 2024 | 3,652 Days | Daily Aggregates | Observational Atmospheric Archive |
| **`weather_delhi_10yr.csv`** | Delhi NCR Region, India (28.61°N, 77.20°E) | Jan 1, 2015 – Dec 31, 2024 | 3,652 Days | Daily Aggregates | IMD / Open-Meteo Reanalysis API |
| **`weather_seattle_benchmark.csv`** | Seattle-Tacoma Airport, WA, USA (47.45°N, 122.30°W) | Jan 1, 2012 – Dec 31, 2015 | 1,461 Days | Daily Station Records | NOAA National Centers for Environmental Information / Vega |
| **`weather_multicity_combined.csv`** | Dual-Region India (Pune + Delhi) | 2015 – 2024 (10 Years) | 7,304 Days | Multi-City Daily Data | Synthesized Spatial Observational Archive |

---

## 2. Feature Dictionary & Measurement Units

| Variable Name | Data Type | Physical Dimension | Physical Description & Atmospheric Significance |
|---|---|---|---|
| `Date` | String (`YYYY-MM-DD`) | Chronological Index | Timestamp of meteorological measurement |
| `MaxTemp` | Continuous Float | Degrees Celsius (°C) | Maximum surface air temperature recorded at 2 meters above ground level |
| `MinTemp` | Continuous Float | Degrees Celsius (°C) | Minimum surface air temperature recorded at 2 meters |
| `TempRange` | Continuous Float | Degrees Celsius (°C) | Diurnal temperature range: $T_{range} = MaxTemp - MinTemp$ (inversely correlated with cloud cover) |
| `Precipitation` | Continuous Float | Millimeters (mm) | Daily accumulated liquid equivalent rainfall depth |
| `Humidity` | Continuous Float | Percent (%) | Daily mean relative humidity: ratio of actual vapor pressure to saturation vapor pressure |
| `Pressure` | Continuous Float | Hectopascals (hPa) | Daily mean atmospheric pressure at local surface elevation |
| `WindSpeed` | Continuous Float | Kilometers/hour (km/h) | Maximum sustained wind gust velocity measured at 10 meters elevation |
| `RainToday` | Binary Discrete ({0, 1}) | Unitless Indicator | 1 if $Precipitation \ge 1.0\text{ mm}$, 0 otherwise |
| `RainTomorrow` | Binary Discrete ({0, 1}) | Target Event | Next day's precipitation occurrence: $RainToday_{t+1}$ |

---

## 3. Categorical Discretization for Empirical Conditional Probability

To compute empirical discrete joint and conditional probability distributions without parametric assumptions, continuous features are partitioned into mutually exclusive and collectively exhaustive bins:

1. **Atmospheric Relative Humidity ($H$)**:
   - $\text{Low}$: $H < 60\%$ (Dry air, low moisture saturation)
   - $\text{Medium}$: $60\% \le H \le 80\%$ (Moderate maritime/monsoon humidity)
   - $\text{High}$: $H > 80\%$ (Near-saturation boundary layer, high convective potential)

2. **Barometric Surface Pressure ($P$)**:
   - $\text{Low}$: $P < 1008\text{ hPa}$ (Cyclonic low-pressure depression, cloud formation)
   - $\text{Normal}$: $1008\text{ hPa} \le P \le 1014\text{ hPa}$ (Standard seasonal pressure)
   - $\text{High}$: $P > 1014\text{ hPa}$ (Anticyclonic high-pressure ridge, clear skies)

3. **Wind Gust Speed ($W$)**:
   - $\text{Calm}$: $W < 15\text{ km/h}$
   - $\text{Moderate}$: $15\text{ km/h} \le W \le 28\text{ km/h}$
   - $\text{Gusty}$: $W > 28\text{ km/h}$

---

## 4. Academic Citations for Datasets
1. Open-Meteo Weather API: Zippenfenig, P. (2023). Open-Meteo: Open-source weather API and historical reanalysis archive. Available at: https://open-meteo.com/
2. NOAA NCEI Daily Global Historical Climatology Network (GHCN-Daily): Menne, M. J., et al. (2012). An overview of the global historical climatology network-daily database. Journal of Atmospheric and Oceanic Technology, 29(7), 897-910.
3. Seattle Weather Dataset: Vega Datasets Repository (2016). Seattle Daily Precipitation and Temperature Series. Available at: https://github.com/vega/vega-datasets
