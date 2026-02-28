import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Date, Numeric, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from edgar import Company, set_identity
import pandas as pd
from datetime import date, datetime
import requests

# SEC EDGAR User-Agent
set_identity("Chandra Shukla shuklach@outlook.com")

# ---------- DB setup ----------
Base = declarative_base()
DATABASE_URL = "sqlite:///./13f.db"
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine)


class Filer(Base):
    __tablename__ = "filers"
    id = Column(Integer, primary_key=True)
    cik = Column(String(20), unique=True, nullable=False)
    name = Column(String, nullable=False)
    filings = relationship("Filing", back_populates="filer")


class Filing(Base):
    __tablename__ = "filings"
    id = Column(Integer, primary_key=True)
    filer_id = Column(Integer, ForeignKey("filers.id"))
    accession_no = Column(String(50), unique=True, nullable=False)
    period_end = Column(Date, nullable=False)
    filed_at = Column(Date, nullable=False)
    filer = relationship("Filer", back_populates="filings")
    holdings = relationship("Holding", back_populates="filing")


class Holding(Base):
    __tablename__ = "holdings"
    id = Column(Integer, primary_key=True)
    filing_id = Column(Integer, ForeignKey("filings.id"))
    cusip = Column(String(12))
    ticker = Column(String(16))
    issuer_name = Column(String)
    shares = Column(Numeric)
    market_value = Column(Numeric)
    filing = relationship("Filing", back_populates="holdings")


Base.metadata.create_all(engine)


# ---------- Utility ----------
def to_date(val):
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        return datetime.strptime(val[:10], "%Y-%m-%d").date()
    return val


# ---------- CUSIP → Ticker lookup via SEC EDGAR ----------
@st.cache_data(show_spinner=False)
def cusip_to_ticker(cusip: str) -> str:
    """Try to resolve CUSIP to ticker via SEC EDGAR company tickers JSON."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": "Chandra Shukla shuklach@outlook.com"}
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        # The JSON maps index → {cik_str, ticker, title}
        # We can't reverse CUSIP from this directly, so return cusip as fallback
        return cusip  # fallback; EDGAR doesn't have CUSIP→ticker in free API
    except Exception:
        return cusip


def get_session():
    return SessionLocal()


def get_or_create_filer(session, cik: str):
    filer = session.query(Filer).filter_by(cik=cik).one_or_none()
    if filer:
        return filer
    company = Company(cik)
    filer = Filer(cik=cik, name=company.name)
    session.add(filer)
    session.commit()
    return filer


def ingest_latest_two_13f(session, cik: str):
    filer = get_or_create_filer(session, cik)
    company = Company(cik)
    filings = company.get_filings(form="13F-HR")
    filings = filings.head(2)
    stored = []

    for filing_obj in filings:
        accession_no = filing_obj.accession_no
        filing_date  = to_date(filing_obj.filing_date)
        report_date  = to_date(filing_obj.report_date)

        existing = session.query(Filing).filter_by(accession_no=accession_no).one_or_none()
        if existing:
            stored.append(existing)
            continue

        thirteen_f = filing_obj.obj()
        if thirteen_f is None:
            continue

        filing = Filing(
            filer_id=filer.id,
            accession_no=accession_no,
            period_end=report_date,
            filed_at=filing_date,
        )
        session.add(filing)
        session.flush()

        infotable = thirteen_f.infotable
        if hasattr(infotable, "to_dataframe"):
            df = infotable.to_dataframe()
        elif hasattr(infotable, "itertuples"):
            df = infotable
        else:
            df = pd.DataFrame(infotable)

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        for row in df.itertuples(index=False):
            cusip        = getattr(row, "cusip", None)
            ticker       = getattr(row, "ticker", None)
            issuer_name  = getattr(row, "name", None) or getattr(row, "nameofissuer", None)
            shares       = getattr(row, "shares", None) or getattr(row, "sshprnamt", None)
            market_value = getattr(row, "value", None)

            h = Holding(
                filing_id=filing.id,
                cusip=str(cusip).strip() if cusip else None,
                ticker=str(ticker).strip() if ticker else None,
                issuer_name=str(issuer_name).strip() if issuer_name else None,
                shares=float(shares) if shares else None,
                market_value=float(market_value) if market_value else None,
            )
            session.add(h)

        stored.append(filing)

    session.commit()
    return sorted(stored, key=lambda x: x.period_end, reverse=True)[:2]


def get_last_two_filings(session, filer_id):
    return (
        session.query(Filing)
        .filter_by(filer_id=filer_id)
        .order_by(Filing.period_end.desc())
        .limit(2)
        .all()
    )


def get_holdings_map(session, filing_id):
    holdings = session.query(Holding).filter_by(filing_id=filing_id).all()
    m = {}
    for h in holdings:
        key = h.cusip or h.ticker
        if not key:
            continue
        m[key] = h
    return m


def compute_activity(session, filer):
    filings = get_last_two_filings(session, filer.id)
    if len(filings) < 2:
        return None
    latest, prev = filings[0], filings[1]
    latest_h = get_holdings_map(session, latest.id)
    prev_h   = get_holdings_map(session, prev.id)

    new_buys, increases, decreases, exits = [], [], [], []
    keys = set(latest_h.keys()) | set(prev_h.keys())

    for key in keys:
        l = latest_h.get(key)
        p = prev_h.get(key)

        # Display ticker if available, else company name, else CUSIP
        display = (l or p)
        label = display.ticker if (display.ticker and display.ticker.strip() and display.ticker != "nan") else                 (display.issuer_name if display.issuer_name else key)
        company_name = display.issuer_name or ""

        if l and not p:
            new_buys.append({
                "Ticker / Name": label,
                "Company": company_name,
                "Action": "🟢 NEW BUY",
                "Prev Shares": 0,
                "New Shares": int(float(l.shares or 0)),
                "Prev Value": "$0",
                "New Value": f"${float(l.market_value or 0):,.0f}K",
                "New Value Raw": float(l.market_value or 0),
                "Change": "↑ New Position",
            })
        elif p and not l:
            exits.append({
                "Ticker / Name": label,
                "Company": company_name,
                "Action": "🔴 SOLD OUT",
                "Prev Shares": int(float(p.shares or 0)),
                "New Shares": 0,
                "Prev Value": f"${float(p.market_value or 0):,.0f}K",
                "New Value": "$0",
                "New Value Raw": 0,
                "Change": "↓ Exited",
            })
        elif l and p:
            ls, ps = float(l.shares or 0), float(p.shares or 0)
            lv, pv = float(l.market_value or 0), float(p.market_value or 0)
            pct = ((ls - ps) / ps * 100) if ps else 0
            if ls > ps:
                increases.append({
                    "Ticker / Name": label,
                    "Company": company_name,
                    "Action": "🔼 INCREASED",
                    "Prev Shares": int(ps),
                    "New Shares": int(ls),
                    "Prev Value": f"${pv:,.0f}K",
                    "New Value": f"${lv:,.0f}K",
                    "New Value Raw": lv,
                    "Change": f"+{pct:.1f}%",
                })
            elif ls < ps:
                decreases.append({
                    "Ticker / Name": label,
                    "Company": company_name,
                    "Action": "🔽 REDUCED",
                    "Prev Shares": int(ps),
                    "New Shares": int(ls),
                    "Prev Value": f"${pv:,.0f}K",
                    "New Value": f"${lv:,.0f}K",
                    "New Value Raw": lv,
                    "Change": f"{pct:.1f}%",
                })

    def make_df(rows):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.sort_values("New Value Raw", ascending=False).drop(columns=["New Value Raw"])
        return df

    return {
        "latest_period": latest.period_end,
        "prev_period":   prev.period_end,
        "new_buys":  make_df(new_buys),
        "increases": make_df(increases),
        "decreases": make_df(decreases),
        "exits":     make_df(exits),
    }


# ---------- Streamlit UI ----------
st.set_page_config(page_title="13F Institutional Activity", layout="wide", page_icon="📊")

st.markdown("""
<style>
/* Dark Bloomberg-style theme */
html, body, [class*="css"] {
    background-color: #0a0e17 !important;
    color: #e0e0e0 !important;
    font-family: 'Courier New', monospace !important;
}
.stApp { background-color: #0a0e17; }
h1 { color: #00d4ff !important; font-size: 1.6rem !important; letter-spacing: 2px; }
h2, h3 { color: #00d4ff !important; }
.stMetric { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 12px; }
.stMetric label { color: #9ca3af !important; font-size: 0.75rem; letter-spacing: 1px; }
.stMetric [data-testid="metric-container"] { color: #ffffff; }
div[data-testid="stDataFrame"] { border: 1px solid #1f2937; border-radius: 6px; }
.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    color: #9ca3af !important; font-weight: bold;
    letter-spacing: 1px; font-size: 0.8rem;
}
.stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff; }
.stSelectbox label, .stNumberInput label { color: #9ca3af !important; font-size: 0.75rem; letter-spacing: 1px; }
.stButton button {
    background: #00d4ff !important; color: #000 !important;
    font-weight: bold !important; letter-spacing: 2px !important;
    border-radius: 4px !important; border: none !important;
}
.stButton button:hover { background: #00a8cc !important; }
.stAlert { border-radius: 6px; }
hr { border-color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## 📊 INSTITUTIONAL 13F ACTIVITY TERMINAL")
    st.caption("SEC EDGAR · Form 13F · Family Offices & Hedge Funds · Quarter-over-Quarter Changes")
with col_h2:
    st.markdown(f"<div style='text-align:right; color:#9ca3af; font-size:0.75rem; margin-top:20px'>"
                f"DATA: SEC EDGAR<br>FORM: 13F-HR<br>LIVE PULL</div>", unsafe_allow_html=True)

st.markdown("---")

# ── Fund selector ────────────────────────────────────────────────────────────
PRESET_FUNDS = {
    "DUQUESNE FAMILY OFFICE  |  Druckenmiller": "0001392245",
    "BERKSHIRE HATHAWAY  |  Buffett":           "0001067983",
    "PERSHING SQUARE  |  Ackman":               "0001336528",
    "TIGER GLOBAL  |  Chase Coleman":           "0001167483",
    "APPALOOSA MANAGEMENT  |  Tepper":          "0000813672",
    "── CUSTOM CIK ──":                         "custom",
}

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    selected = st.selectbox("SELECT FUND", list(PRESET_FUNDS.keys()))
with col2:
    min_value = st.number_input("MIN VALUE ($K)", min_value=0, value=0, step=1000)
with col3:
    st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
    load = st.button("▶  LOAD DATA", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if PRESET_FUNDS[selected] == "custom":
    cik = st.text_input("ENTER CIK", value="", placeholder="e.g. 0001392245")
else:
    cik = PRESET_FUNDS[selected]

if load:
    if not cik:
        st.error("Please enter a CIK.")
    else:
        with st.spinner("PULLING 13F FILINGS FROM SEC EDGAR..."):
            session = get_session()
            try:
                ingest_latest_two_13f(session, cik)
                filer = session.query(Filer).filter_by(cik=cik).one_or_none()
                if not filer:
                    st.error("Filer not found.")
                else:
                    activity = compute_activity(session, filer)
                    if not activity:
                        st.warning("Not enough filings to compare.")
                    else:
                        # ── Summary bar ──────────────────────────────────
                        st.markdown("---")
                        st.markdown(f"### {filer.name.upper()}")
                        st.caption(f"PERIOD: {activity['prev_period']}  →  {activity['latest_period']}   |   CIK: {cik}")
                        st.markdown("---")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("🟢 NEW BUYS",  len(activity["new_buys"]))
                        m2.metric("🔼 INCREASED", len(activity["increases"]))
                        m3.metric("🔽 REDUCED",   len(activity["decreases"]))
                        m4.metric("🔴 EXITS",     len(activity["exits"]))

                        st.markdown("---")

                        def show_table(df, min_val):
                            if df.empty:
                                st.info("No activity in this category.")
                                return
                            display_cols = ["Ticker / Name", "Company", "Action",
                                            "Prev Shares", "New Shares",
                                            "Prev Value", "New Value", "Change"]
                            df_show = df[display_cols] if all(c in df.columns for c in display_cols) else df
                            st.dataframe(
                                df_show,
                                use_container_width=True,
                                hide_index=True,
                                height=min(600, 50 + 40 * len(df_show)),
                            )

                        tab1, tab2, tab3, tab4 = st.tabs([
                            f"🟢  NEW BUYS  ({len(activity['new_buys'])})",
                            f"🔼  INCREASED  ({len(activity['increases'])})",
                            f"🔽  REDUCED  ({len(activity['decreases'])})",
                            f"🔴  EXITED  ({len(activity['exits'])})",
                        ])

                        with tab1:
                            show_table(activity["new_buys"], min_value)
                        with tab2:
                            show_table(activity["increases"], min_value)
                        with tab3:
                            show_table(activity["decreases"], min_value)
                        with tab4:
                            show_table(activity["exits"], min_value)

            except Exception as e:
                st.error(f"Error fetching data: {e}")
            finally:
                session.close()
