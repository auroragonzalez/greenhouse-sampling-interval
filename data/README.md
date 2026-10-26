# Data

**Status: the real campaign data are being prepared for publication on Zenodo
and are not available yet.** This folder will link the Zenodo record (DOI)
once it is published.

Until then, `make data` (`python3 tools/make_synthetic_data.py`) fills this
folder with **synthetic, randomly generated** files. They mimic the format of
the real exports, and a `SYNTHETIC_DATA_README.txt` file marks the folder as
synthetic. The values come from a toy first-order plant with
random weather. The schedule follows the protocol, but the values are **not
measurements**, and analyses run on them do not reproduce the paper.

## Expected layout

```
data/
  raw/                                  one CSV per channel: _time (UTC, ISO 8601), _value
    Temperatura_Interior_{Alta,Centro,Baja}_(ºC)_0.csv
    Humedad_Relativa_interior_{Alta,Centro,Baja}_(%)_0.csv
    CO2_(ppm)_0.csv, DPV_(kPa)_0.csv
    PAR_interior_(µmol_m_-2_s_-1)_0.csv, Radiación_Solar_Global_Interior_(W_m_-2)_0.csv
    Ventana_cenital_{este,oeste}_0.csv                 roof vents, aperture %
    Ventana_lateral_{norte,sur,oeste}_0.csv            side vents, aperture %
    Pantalla_de_sombreo_cenital_0.csv                  shade screen, %
    Luz_artificial_<position>_0.csv  (6 lamps)         intensity 0-10
    Fog_0.csv                                          valve on/off events
    Recirculador_0.csv, Recirculador_centro_0.csv      recirculator 0/1
    Calefactores_0.csv, Calefactores_2.csv             heaters used as recirculators 0/1
    outside data/                       on-site weather station
      Temperatura_ambiente.csv, Humedad_ambiente.csv,
      Radiacion_solar_par.csv, Velocidad_del_viento.csv,
      Radiación_solar_global_0.csv (dead channel), Precipitacion.csv
    forecast data/hourly_weather_data.csv   Open-Meteo, local time
  openmeteo/                            Open-Meteo, local time
    openmeteo_archive_2026-08-01_2026-09-23.csv
    openmeteo_histforecast_2026-08-01_2026-09-23.csv
    openmeteo_dayahead_2026-08-01_2026-09-23.csv
  blocks/august_blocks.csv              commanded level (cmd_<GROUP>) per 3 h August block
```

Negative values in actuator files are status codes, not levels. The decoding
rules (time zone, fog duty cycle, the device files that make up each group)
are documented in `src/protocol_check.py`.
