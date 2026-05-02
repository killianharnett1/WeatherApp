# Ireland Climate Explorer 

## Overview
The Ireland Climate Explorer is an interactive multi-page dashboard built using **Shiny for Python**. It allows users to explore historical climate patterns across Ireland, focusing on rainfall and temperature data from multiple weather stations.

The application is designed for general users to investigate climate trends through interactive visualisations, comparisons, and insights.

---

## Live Application
 **Access the dashboard here:**  
[https://019dca9f-395b-968d-8266-f6bb4a660238.share.connect.posit.cloud/]


---

## Features

### Overview
- Compare annual rainfall and temperature across stations  
- KPI cards and quick insights  
- Interactive station map  

### Trends
- Time-series visualisations  
- Multi-station comparison  
- Summary statistics and downloadable table  

### Seasonality
- Average monthly climate patterns  
- Peak and low month identification  
- Seasonal range comparison  

### Anomalies
- Monthly deviation from long-term averages  
- Identification of extreme months  
- Summary insights  

---

## Dataset

The dataset consists of historical monthly climate observations for five Irish weather stations:

- Dublin Airport  
- Cork Airport  
- Shannon Airport  
- Athenry  
- Belmullet  

Data includes:
- Rainfall (mm)  
- Temperature (°C)  

The application loads data from a GitHub-hosted CSV file, with a local fallback included.

---

## Technologies Used

- Python  
- Shiny for Python  
- Plotly  
- Pandas  
- NumPy  

---

## Installation (Run Locally)

1. Clone the repository:
```bash
git clone https://github.com/killianharnett1/WeatherApp.git
cd WeatherApp
