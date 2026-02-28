import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Date, Numeric, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from edgar import Company, set_identity
import pandas as pd
from datetime import date, datetime

# SEC EDGAR requires a User-Agent identity
set_identity("Chandra Shukla shuklach@outlook.com")

# ---------- DB setup (SQLite) ----------
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
    """Convert string, datetime, or date to Python date object."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        return datetime.strptime(val[:10], "%Y-%m-%d").date()
    return val


# ---------- Helper functions ----------
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

        # infotable is a DataFrame in current edgartools versions
        # columns: name, cusip, ticker, value, shares, type, etc.
        infotable = thirteen_f.infotable
        if hasattr(infotable, "to_dataframe"):
            df = infotable.to_dataframe()
        elif hasattr(infotable, "itertuples"):
            df = infotable  # already a DataFrame
        else:
            df = pd.DataFrame(infotable)

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        for row in df.itertuples(index=False):
            cusip        = getattr(row, "cusip",  None)
            ticker       = getattr(row, "ticker", None)
            issuer_name  = getattr(row, "name",   None)
            shares       = getattr(row, "shares", None) or getattr(row, "sshprnamt", None)
            market_value = getattr(row, "value",  None)

            h = Holding(
                filing_id=filing.id,
                cusip=str(cusip) if cusip else None,
                ticker=str(ticker) if ticker else None,
                issuer_name=str(issuer_name) if issuer_name else None,
                shares=float(shares) if shares else None,
                market_value=float(market_value) if market_value else None,
            )
            session.add(h)

        stored.append(filing)

    session.commit()
    return sorted(stored, key=lambda x: x.period_end, reverse=True)[:2]


def get_last_two_filings(session, filer_id: int):
    return (
        session.query(Filing)
        .filter_by(filer_id=filer_id)
        .order_by(Filing.period_end.desc())
        .limit(2)
        .all()
    )


def get_holdings_map(session, filing_id: int):
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

        if l and not p:
            new_buys.append({
                "Instrument": key, "Issuer": l.issuer_name,
                "Prev Shares": 0, "New Shares": float(l.shares or 0),
                "Prev Value ($)": 0, "New Value ($)": float(l.market_value or 0),
            })
        elif p and not l:
            exits.append({
                "Instrument": key, "Issuer": p.issuer_name,
                "Prev Shares": float(p.shares or 0), "New Shares": 0,
                "Prev Value ($)": float(p.market_value or 0), "New Value ($)": 0,
            })
        elif l and p:
            ls, ps = float(l.shares or 0), float(p.shares or 0)
            lv, pv = float(l.market_value or 0), float(p.market_value or 0)
            row = {
                "Instrument": key, "Issuer": l.issuer_name,
                "Prev Shares": ps, "New Shares": ls,
                "Prev Value ($)": pv, "New Value ($)": lv,
            }
            if ls > ps:
                increases.append(row)
            elif ls < ps:
                decreases.append(row)

    return {
        "latest_period": latest.period_end,
        "prev_period":   prev.period_end,
        "new_buys":   pd.DataFrame(new_buys),
        "increases":  pd.DataFrame(increases),
        "decreases":  pd.DataFrame(decreases),
        "exits":      pd.DataFrame(exits),
    }


# ---------- Streamlit UI ----------
st.set_page_config(page_title="13F Activity Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #00d4ff; }
    h2, h3 { color: #ffffff; }
    .stTabs [data-baseweb="tab"] { color: #ffffff; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 13F Recent Buys & Sells Dashboard")
st.caption("Powered by SEC EDGAR | Family Office & Institutional Holdings")

PRESET_FUNDS = {
    "Duquesne Family Office (Druckenmiller)": "0001392245",
    "Berkshire Hathaway": "0001067983",
    "Pershing Square (Ackman)": "0001336528",
    "Tiger Global": "0001167483",
    "Appaloosa Management (Tepper)": "0000813672",
    "Custom CIK": "custom",
}

col1, col2 = st.columns([2, 1])
with col1:
    selected = st.selectbox("Select a fund or enter custom CIK", list(PRESET_FUNDS.keys()))
with col2:
    min_value = st.number_input("Min position value ($)", min_value=0, value=0, step=100000)

if PRESET_FUNDS[selected] == "custom":
    cik = st.text_input("Enter CIK manually", value="")
else:
    cik = PRESET_FUNDS[selected]
    st.info(f"CIK: `{cik}`")

if st.button("🔍 Load 13F Activity", use_container_width=True):
    if not cik:
        st.error("Please enter a CIK.")
    else:
        with st.spinner("Fetching latest 13F filings from SEC EDGAR..."):
            session = get_session()
            try:
                ingest_latest_two_13f(session, cik)
                filer = session.query(Filer).filter_by(cik=cik).one_or_none()
                if not filer:
                    st.error("Filer not found.")
                else:
                    activity = compute_activity(session, filer)
                    if not activity:
                        st.warning("Not enough filings found to compare.")
                    else:
                        st.success(f"Loaded: {filer.name}")
                        st.subheader(f"{filer.name}")
                        st.caption(
                            f"Comparing **{activity['prev_period']}** → **{activity['latest_period']}**"
                        )

                        def filter_df(df):
                            if df.empty or min_value == 0:
                                return df
                            return df[df["New Value ($)"] >= min_value]

                        tab1, tab2, tab3, tab4 = st.tabs([
                            "🟢 New Buys", "🔼 Increases", "🔽 Decreases", "🔴 Exits"
                        ])

                        with tab1:
                            df = filter_df(activity["new_buys"])
                            st.metric("Total New Buys", len(df))
                            st.dataframe(df.sort_values("New Value ($)", ascending=False), use_container_width=True)
                            if not df.empty:
                                st.bar_chart(df.set_index("Instrument")["New Value ($)"].head(10))

                        with tab2:
                            df = filter_df(activity["increases"])
                            st.metric("Total Increases", len(df))
                            st.dataframe(df.sort_values("New Value ($)", ascending=False), use_container_width=True)

                        with tab3:
                            df = filter_df(activity["decreases"])
                            st.metric("Total Decreases", len(df))
                            st.dataframe(df.sort_values("Prev Value ($)", ascending=False), use_container_width=True)

                        with tab4:
                            df = filter_df(activity["exits"])
                            st.metric("Total Exits", len(df))
                            st.dataframe(df.sort_values("Prev Value ($)", ascending=False), use_container_width=True)

            except Exception as e:
                st.error(f"Error fetching data: {e}")
            finally:
                session.close()
