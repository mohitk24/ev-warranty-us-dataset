# 🔋 US Battery Electric Vehicles Dataset
US Electric Vehicle Battery Warranty &amp; Specifications Database (2010–2025)

A structured, source-cited dataset of battery warranty terms and technical 
specifications for electric vehicles sold in the United States — compiled to 
make information that is normally scattered across dozens of manufacturer 
websites accessible in a single, analysis-ready format.

---

## 📖 Overview

This dataset covers **1,117 model-year entries** from **34 manufacturers** across 
model years **2010 to 2025**, including both Battery Electric Vehicles (BEV) and 
Plug-in Hybrid Electric Vehicles (PHEV). Every warranty figure is traceable back 
to an official manufacturer source or a verified technical reference, with URLs 
included directly in the dataset for transparency and reproducibility.

---

## 📁 File

| File | Description |
|------|-------------|
| `EV_Warranty_US.csv` | Main dataset — 1,117 rows, 19 columns |

---

## 🗂️ Column Reference

| Column | Description |
|--------|-------------|
| `Manufacturer` | Brand name (e.g., Tesla, Ford, BMW) |
| `EV Model` | Model name |
| `Model Version` | Trim or variant |
| `Wheel-Drive` | Rear / Front / All (AWD) |
| `EV Type` | BEV (Battery Electric) or PHEV (Plug-in Hybrid) |
| `EV Year` | Model year (2010–2025) |
| `Warranty - length in years` | Duration of the battery warranty |
| `Warranty - battery health/capacity retention` | Minimum retained capacity guaranteed (%) |
| `Warranty - mileage` | Mileage cap for warranty coverage |
| `Battery Chemistry` | Cell chemistry type — NMC, LFP, NCA, NCMA, or LMO |
| `Battery capacity (kWh)` | Total (gross) battery pack capacity |
| `Useable capacity (kWh)` | Usable energy capacity — some missing values for earlier years |
| `Charge Power max (kW)` | Maximum DC fast-charge rate supported |
| `Battery Warranty Sources` | URL to the official manufacturer warranty page |
| `Chemistry Sources` | URL to the technical reference for battery chemistry |

> ⚠️ **Note:** The `Features` column is sparsely populated and may be safely 
> dropped for most analyses. Usable capacity values have some missing entries, 
> particularly for model years prior to 2015. All warranty figures reflect 
> **US market terms** and may differ from international variants.

---

## 💡 Use Cases

This dataset is well-suited for a wide range of analytical, research, and 
educational purposes, including cross-brand warranty comparison, studying how 
battery chemistry correlates with warranty generosity, analyzing BEV vs. PHEV 
coverage differences, tracking how EV specifications have evolved over the 
15-year span, building EV recommendation tools or buyer guides, and supporting 
academic or policy research on EV market maturity.

---

## 🏭 Manufacturers Covered

Audi · BMW · Cadillac · Chevrolet · Fiat · Fisker · Ford · GMC · Genesis · 
Honda · Hyundai · Jaguar · Jeep · KIA · Lexus · Lotus · Lucid · MINI · 
Maserati · Mazda · Mercedes-Benz · Nissan · Polestar · Porsche · Ram · 
Rivian · Rolls Royce · Subaru · Tesla · Toyota · VinFast · Volkswagen · 
Volvo · smart

---

## 🔍 Data Sources

All data was compiled from official manufacturer warranty pages and Wikipedia 
technical references. Source URLs are included per row in the dataset for 
independent verification.

---

## 📜 License

This dataset is released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license. You are free to share and adapt it for any purpose, provided you give 
appropriate credit to the original author.

---

## 🙌 Acknowledgements

If you use this dataset in your work, a citation or a link back to this 
repository is appreciated. Contributions, corrections, and pull requests 
are welcome!
