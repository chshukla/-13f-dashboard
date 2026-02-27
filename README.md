# 13F Activity Dashboard

An interactive Streamlit dashboard that pulls SEC 13F filings for major family offices and institutions, and shows recent **buys, increases, decreases, and exits** between the last two quarters.

## Features
- Preset list of well-known funds (Druckenmiller, Berkshire, Ackman, Tiger Global, etc.)
- Custom CIK entry for any 13F filer
- Filter by minimum position value
- Four tabs: New Buys | Increases | Decreases | Exits
- Bar chart for top new buys

## How to run locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`

## Data source
All data is pulled directly from **SEC EDGAR** via [edgartools](https://github.com/dgunning/edgartools). No paid API required.

## Deploy free on Streamlit Community Cloud
1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo and select `app.py`
4. Click Deploy
