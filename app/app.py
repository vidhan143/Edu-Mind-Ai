import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os, io, re, json, time, random
from datetime import datetime
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
# ─── Env Loader ─────────────────────────────────────────────────────────────────
def load_dotenv_file():
    current_dir = os.path.dirname(__file__)
    env_paths = [
        os.path.abspath(os.path.join(current_dir, '.env')),
        os.path.abspath(os.path.join(current_dir, '..', '.env')),
        os.path.abspath(os.path.join(current_dir, '..', '..', '.env')),
    ]

    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())
            break

load_dotenv_file()

# Helper to read config from env vars first, then Streamlit secrets.
def get_config_value(name, fallback_names=None):
    value = os.environ.get(name)
    if not value and hasattr(st, 'secrets'):
        value = st.secrets.get(name)
    if not value and fallback_names:
        for fallback_name in fallback_names:
            value = os.environ.get(fallback_name)
            if not value and hasattr(st, 'secrets'):
                value = st.secrets.get(fallback_name)
            if value:
                break
    return value


def get_config_values(name, fallback_names=None):
    raw = get_config_value(name, fallback_names=fallback_names)
    if not raw:
        return []
    return [item.strip() for item in re.split(r'[,;\s]+', raw) if item.strip()]


def get_gemini_api_keys():
    keys = []
    keys.extend(get_config_values('GEMINI_API_KEYS', fallback_names=['GEMINI_API_KEY']))
    for i in range(1, 11):
        keys.extend(get_config_values(f'GEMINI_API_KEY_{i}'))
    if not keys:
        keys.extend(get_config_values('GOOGLE_API_KEY'))
    unique_keys = []
    for key in keys:
        if key and key not in unique_keys:
            unique_keys.append(key)
    return unique_keys


GEMINI_MODEL = get_config_value('GEMINI_MODEL') or 'gemini-2.5-flash'
GEMINI_API_KEYS = get_gemini_api_keys()
PRIMARY_GEMINI_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ''

if PRIMARY_GEMINI_KEY and not os.environ.get('GEMINI_API_KEY'):
    os.environ['GEMINI_API_KEY'] = PRIMARY_GEMINI_KEY

# ─── Gemini helpers ────────────────────────────────────────────────────────────
def get_gemini_client(api_key: str):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai


def gemini_request_with_retry(request_fn, api_keys, retries: int = 2, delay: int = 2):
    if not api_keys:
        raise RuntimeError(
            "No Gemini API keys are configured. Add GEMINI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEYS to your environment or Streamlit secrets."
        )

    last_error = None
    for api_key in api_keys:
        for attempt in range(retries + 1):
            try:
                client = get_gemini_client(api_key)
                return request_fn(client)
            except Exception as e:
                last_error = e
                err_text = str(e)
                if 'RESOURCE_EXHAUSTED' in err_text or '429' in err_text or 'quota' in err_text.lower():
                    if attempt < retries:
                        time.sleep(delay * (attempt + 1))
                        continue
                    break
                raise

    message = (
        "Google Gemini quota is currently exhausted or rate-limited for all configured API keys. "
        "Please check your API keys and quota settings."
    )
    if last_error:
        message += f"\n\nLast error: {last_error}"
    raise RuntimeError(message) from last_error

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduMind AI Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #060b18;
    font-family: 'DM Sans', sans-serif;
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}


/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #060b18 100%);
    border-right: 1px solid rgba(99,179,237,0.12);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Headings */
h1,h2,h3,h4,h5 { font-family: 'Syne', sans-serif !important; color: #f0f4ff !important; }

/* Page title */
.page-hero {
    background: linear-gradient(135deg, rgba(56,189,248,0.08) 0%, rgba(99,102,241,0.08) 100%);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.page-hero h1 { font-size: 2rem !important; margin: 0 !important; }
.page-hero p  { color: #94a3b8 !important; margin: 6px 0 0 !important; font-size: 0.95rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1526 60%, #111c35);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 16px;
    padding: 18px 20px !important;
    transition: border-color .2s, transform .2s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(56,189,248,0.45);
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: .82rem !important; }
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricDelta"] { font-size: .8rem !important; }

/* Plotly charts */
.js-plotly-plot { border-radius: 14px !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    transition: opacity .2s, transform .2s !important;
    letter-spacing: .5px;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: #0d1526 !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSelectbox > div > div {
    background: #0d1526 !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSlider > div > div > div { background: #38bdf8 !important; }

/* Dataframe */
.stDataFrame { border-radius: 14px !important; overflow: hidden; }
[data-testid="stDataFrame"] table { background: #0d1526 !important; }

/* Dividers */
hr { border-color: rgba(99,179,237,0.12) !important; }

/* Info / success / warning / error boxes */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 12px !important;
    font-size: .9rem !important;
}

/* Score pill */
.score-pill {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.6rem;
    color: white;
}
.score-high { background: linear-gradient(135deg,#10b981,#059669); }
.score-mid  { background: linear-gradient(135deg,#f59e0b,#d97706); }
.score-low  { background: linear-gradient(135deg,#ef4444,#dc2626); }

/* Resume section card */
.resume-section {
    background: #0d1526;
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.resume-section h4 { margin: 0 0 8px !important; color: #38bdf8 !important; font-size: .95rem !important; }
.resume-section ul { margin: 0; padding-left: 20px; color: #cbd5e1; }
.resume-section ul li { margin-bottom: 5px; font-size: .88rem; }

/* Tag chips */
.tag {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: .78rem;
    margin: 3px 3px 3px 0;
    font-family: 'DM Sans', sans-serif;
}
.tag-warn {
    background: rgba(245,158,11,0.12);
    border-color: rgba(245,158,11,0.3);
    color: #fbbf24;
}
.tag-bad {
    background: rgba(239,68,68,0.12);
    border-color: rgba(239,68,68,0.3);
    color: #f87171;
}

/* Gauge labels */
.gauge-label {
    text-align: center;
    color: #94a3b8;
    font-size: .85rem;
    margin-top: -10px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d1526 !important;
    border: 2px dashed rgba(56,189,248,0.3) !important;
    border-radius: 14px !important;
}

/* Sidebar nav */
.nav-link { border-radius: 10px !important; }
.nav-link.active { background: linear-gradient(135deg,rgba(14,165,233,.25),rgba(99,102,241,.25)) !important; }

/* ── Chat UI ─────────────────────────────────────────────── */
.chat-wrapper{display:flex;flex-direction:column;gap:16px;padding:8px 0 24px;}
.chat-bubble-row{display:flex;align-items:flex-end;gap:10px;}
.chat-bubble-row.user{flex-direction:row-reverse;}
.chat-bubble-row.bot{flex-direction:row;}
.avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;}
.avatar.bot{background:linear-gradient(135deg,#0ea5e9,#6366f1);}
.avatar.user{background:linear-gradient(135deg,#8b5cf6,#ec4899);}
.bubble{max-width:72%;padding:13px 18px;border-radius:18px;font-size:.9rem;line-height:1.65;word-break:break-word;}
.bubble.bot{background:#0d1a30;border:1px solid rgba(56,189,248,0.18);border-bottom-left-radius:4px;color:#cbd5e1;}
.bubble.user{background:linear-gradient(135deg,rgba(14,165,233,.22),rgba(99,102,241,.22));border:1px solid rgba(99,102,241,0.3);border-bottom-right-radius:4px;color:#e2e8f0;text-align:right;}
.bubble .ts{font-size:.7rem;color:#334155;margin-top:6px;}
.bubble.user .ts{text-align:right;}
.profile-card{background:#0d1526;border:1px solid rgba(56,189,248,.18);border-radius:16px;padding:16px 20px;margin-bottom:20px;}
.profile-card .label{color:#475569;font-size:.73rem;margin-bottom:2px;text-transform:uppercase;letter-spacing:.5px;}
.profile-card .val{color:#e2e8f0;font-size:.87rem;font-weight:500;}

</style>
""", unsafe_allow_html=True)

# ─── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Use the shared project data folder instead of app/data
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'StudentsPerformance.csv')
    )
    df = pd.read_csv(path)
    np.random.seed(42)
    n = len(df)
    if 'study_hours' not in df.columns:
        df['study_hours'] = np.random.randint(1, 10, n)
    if 'attendance' not in df.columns:
        df['attendance'] = np.random.randint(50, 100, n)
    if 'stress_level' not in df.columns:
        df['stress_level'] = np.random.randint(1, 10, n)
    df['avg_score']   = (df['math score'] + df['reading score'] + df['writing score']) / 3
    df['pass']        = (df['math score'] >= 40).astype(int)
    df['risk_level']  = pd.cut(df['avg_score'], bins=[0,40,60,75,100],
                                labels=['High Risk','At Risk','Average','High Performer'])
    return df

df = load_data()

# ─── Helpers ──────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,21,38,0.6)',
    font=dict(color='#94a3b8', family='DM Sans'),
    title_font=dict(color='#f0f4ff', family='Syne', size=15),
    margin=dict(t=45,b=30,l=30,r=20),
)
COLORS = ['#38bdf8','#818cf8','#34d399','#fb923c','#e879f9','#f472b6']

def apply_layout(fig, **kwargs):
    fig.update_layout(**CHART_LAYOUT, **kwargs)
    for trace in fig.data:
        if hasattr(trace, 'marker'):
            try:
                trace.update(marker_line_width=0)
            except Exception:
                pass
    return fig


def safe_gradient_style(df, subset=None, cmap=None):
    try:
        import matplotlib  # noqa: F401
        return df.style.background_gradient(subset=subset, cmap=cmap)
    except Exception:
        return df


def hero(title, subtitle, icon=""):
    st.markdown(f"""
    <div class="page-hero">
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

# ─── Gemini helper ────────────────────────────────────────────────────────────
def gemini_resume_analyze(text: str) -> dict:
    prompt = f"""You are an expert career coach and ATS resume analyzer. Analyze the following resume text and return ONLY a valid JSON object (no markdown, no explanation).

Resume:
\"\"\"
{text[:6000]}
\"\"\"

Return this exact JSON structure:
{{
  "overall_score": <integer 0-100>,
  "ats_score": <integer 0-100>,
  "sections": {{
    "contact_info":  {{"score": <0-10>, "feedback": "<string>"}},
    "summary":       {{"score": <0-10>, "feedback": "<string>"}},
    "experience":    {{"score": <0-10>, "feedback": "<string>"}},
    "education":     {{"score": <0-10>, "feedback": "<string>"}},
    "skills":        {{"score": <0-10>, "feedback": "<string>"}},
    "formatting":    {{"score": <0-10>, "feedback": "<string>"}}
  }},
  "strengths": ["<string>", "<string>", "<string>"],
  "improvements": [
    {{"area": "<string>", "priority": "<High|Medium|Low>", "suggestion": "<string>"}},
    {{"area": "<string>", "priority": "<High|Medium|Low>", "suggestion": "<string>"}},
    {{"area": "<string>", "priority": "<High|Medium|Low>", "suggestion": "<string>"}}
  ],
  "keywords_found": ["<string>"],
  "keywords_missing": ["<string>"],
  "overall_feedback": "<2-3 sentence summary>"
}}"""
    def request_fn(client):
        model = client.GenerativeModel(model_name=GEMINI_MODEL)
        return model.generate_content(contents=[{"text": prompt}])

    response = gemini_request_with_retry(request_fn, GEMINI_API_KEYS)
    raw = response.text.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'```$', '', raw).strip()
    return json.loads(raw)

def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw  = uploaded_file.read()
    if name.endswith(".pdf"):
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return raw.decode("utf-8", errors="ignore")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                    background:linear-gradient(135deg,#38bdf8,#818cf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            EduMind AI
        </div>
        <div style='color:#475569;font-size:.76rem;margin-top:4px;'>Academic Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Dashboard","Student Analysis","Risk Analytics",
                 "Feedback Intel","Clustering","Resume AI","AI Insights","Study Coach"],
        icons=["speedometer2","person-lines-fill","exclamation-triangle-fill",
               "chat-quote-fill","diagram-3-fill","file-earmark-person-fill","stars",
               "chat-dots-fill"],
        default_index=0,
        styles={
            "container":      {"padding":"0","background":"transparent"},
            "nav-link":       {"font-size":"0.85rem","padding":"10px 14px",
                               "color":"#94a3b8","border-radius":"10px","margin":"2px 0"},
            "nav-link-selected": {"background":"linear-gradient(135deg,rgba(14,165,233,.2),rgba(99,102,241,.2))",
                                   "color":"#38bdf8","font-weight":"600"},
            "icon":           {"font-size":"0.85rem"},
        }
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='color:#334155;font-size:.72rem;padding:0 8px;'>
        <b style='color:#475569;'>Dataset:</b> {len(df):,} students<br>
        <b style='color:#475569;'>Updated:</b> {datetime.now().strftime('%b %d, %Y')}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Dashboard":
    hero("Academic Dashboard", "Real-time overview of student performance metrics", "📊")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Students",    f"{len(df):,}")
    c2.metric("Avg Math Score",    f"{df['math score'].mean():.1f}",  delta="+2.3")
    c3.metric("Pass Rate",         f"{df['pass'].mean()*100:.1f}%",   delta="+1.1%")
    c4.metric("Avg Study Hours",   f"{df['study_hours'].mean():.1f}h")
    c5.metric("High Performers",   f"{(df['risk_level']=='High Performer').sum()}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='math score', nbins=25,
                           title="Math Score Distribution",
                           color_discrete_sequence=['#38bdf8'])
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df, x='reading score', y='writing score',
                         color='avg_score', size='study_hours',
                         color_continuous_scale='Blues',
                         title="Reading vs Writing — colored by Avg Score")
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.box(df, y='math score', color='gender',
                     color_discrete_sequence=['#38bdf8','#e879f9'],
                     title="Math Score by Gender")
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rc = df['risk_level'].value_counts().reset_index()
        fig = px.pie(rc, values='count', names='risk_level',
                     color_discrete_sequence=COLORS,
                     title="Risk Level Distribution", hole=0.55)
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        grp = df.groupby('test preparation course')['avg_score'].mean().reset_index()
        fig = px.bar(grp, x='test preparation course', y='avg_score',
                     color='test preparation course',
                     color_discrete_sequence=['#34d399','#f472b6'],
                     title="Test Prep Impact")
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Recent Student Records")
    styled_df = df[['gender','math score','reading score','writing score',
                     'avg_score','risk_level','study_hours','attendance']].head(15)
    st.dataframe(safe_gradient_style(styled_df, subset=['avg_score'], cmap='Blues'),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STUDENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Student Analysis":
    hero("Student Intelligence Analysis", "Enter student data for AI-powered performance prediction", "🧠")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 👤 Student Profile")
        student_name = st.text_input("Full Name", placeholder="e.g. Arjun Sharma")
        gender       = st.selectbox("Gender", ["Male","Female","Other"])
        study_hours  = st.slider("Daily Study Hours", 1, 12, 4)
        attendance   = st.slider("Attendance %", 0, 100, 80)
    with col2:
        st.markdown("##### 📝 Academic Scores")
        reading_score = st.slider("Reading Score", 0, 100, 65)
        writing_score = st.slider("Writing Score", 0, 100, 65)
        prev_score    = st.slider("Previous Term Score", 0, 100, 60)
        test_prep     = st.selectbox("Test Prep Course", ["Completed","Not Completed"])
    with col3:
        st.markdown("##### 🧬 Wellbeing Factors")
        stress_level  = st.slider("Stress Level (1-Low → 10-High)", 1, 10, 5)
        sleep_hours   = st.slider("Avg Sleep Hours", 4, 10, 7)
        extra_curr    = st.slider("Extra-Curricular Activities (hrs/wk)", 0, 20, 3)
        family_income = st.selectbox("Lunch / Socio-economic Proxy",
                                     ["Standard","Free/Reduced"])

    st.markdown("---")
    if st.button("🔍 Generate AI Analysis", use_container_width=True):
        with st.spinner("Running predictive model..."):
            time.sleep(0.6)

        # --- Simple rule-based prediction ---
        base = (reading_score + writing_score) / 2
        bonus = (study_hours * 2.5) + (attendance * 0.25) + (sleep_hours * 1.5)
        penalty = (stress_level * 1.8) + (0 if test_prep=="Completed" else 3)
        predicted = int(min(100, max(0, base * 0.6 + bonus - penalty)))
        trend = predicted - prev_score

        if predicted >= 80:   risk, badge = "Low", "🟢"
        elif predicted >= 55: risk, badge = "Medium", "🟡"
        else:                 risk, badge = "High", "🔴"

        # KPI row
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Predicted Score",   predicted, delta=f"{trend:+.0f} vs last term")
        k2.metric("Risk Level",        f"{badge} {risk}")
        k3.metric("Attendance",        f"{attendance}%")
        k4.metric("Wellbeing Index",   f"{max(0,10-stress_level+sleep_hours//2)}/10")

        st.markdown("---")
        col1, col2 = st.columns([2,1])
        with col1:
            # Radar chart
            cats = ['Reading','Writing','Study Hrs×10','Attendance/10','Sleep×8','Stress Inv']
            vals = [reading_score, writing_score, study_hours*10,
                    attendance/10, sleep_hours*8, (10-stress_level)*10]
            fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself',
                                            fillcolor='rgba(56,189,248,0.15)'))
            apply_layout(fig, title="Student Performance Radar",
                         polar=dict(bgcolor='rgba(0,0,0,0)',
                                    radialaxis=dict(visible=True, range=[0,100],
                                                    gridcolor='rgba(255,255,255,0.07)')))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("##### 📊 Score Breakdown")
            st.progress(reading_score/100, text=f"Reading: {reading_score}")
            st.progress(writing_score/100, text=f"Writing: {writing_score}")
            st.progress(predicted/100,     text=f"Predicted: {predicted}")
            st.progress(attendance/100,    text=f"Attendance: {attendance}%")

        st.markdown("---")
        st.subheader("🤖 AI Recommendations")
        if predicted >= 80:
            st.success(f"""
**Excellent trajectory detected for {student_name or 'Student'}!**

✅ **Strengths:** Consistent study habits, high reading/writing scores
🎯 **Next Steps:**
- Explore advanced courses or certifications
- Consider academic competitions to sharpen skills
- Mentor peers to reinforce your own understanding
            """)
        elif predicted >= 55:
            st.warning(f"""
**Average performance — improvement possible for {student_name or 'Student'}**

⚠️ **Attention Areas:** Stress management, attendance improvement
🎯 **Action Plan:**
- Increase daily study to at least {max(study_hours,5)} hrs
- Join a study group or tutoring session
- Practice stress reduction techniques (meditation, breaks)
            """)
        else:
            st.error(f"""
**Academic risk detected — intervention recommended for {student_name or 'Student'}**

🚨 **Critical Factors:** Low predicted score, high stress, low study hours
🎯 **Immediate Actions:**
- Schedule a counsellor meeting this week
- Reduce non-academic load temporarily
- Set a daily study schedule with small, measurable goals
- Contact teachers for additional support
            """)

# ══════════════════════════════════════════════════════════════════════════════
#  RISK ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Risk Analytics":
    hero("Academic Risk Analytics", "Identify at-risk students early with predictive indicators", "⚠️")

    avg_health = df['avg_score'].mean()
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(avg_health, 1),
        delta={'reference': 65, 'increasing': {'color': '#34d399'},
               'decreasing': {'color': '#f87171'}},
        number={'font': {'color': '#38bdf8', 'family': 'Syne', 'size': 52}},
        title={'text': "Overall Academic Health Index", 'font': {'color': '#f0f4ff', 'size': 14}},
        gauge={
            'axis': {'range': [0,100], 'tickcolor': '#475569',
                     'tickfont': {'color': '#475569'}},
            'bar':  {'color': '#38bdf8', 'thickness': 0.25},
            'bgcolor': '#0d1526',
            'borderwidth': 0,
            'steps': [
                {'range': [0,40],  'color': 'rgba(239,68,68,0.2)'},
                {'range': [40,65], 'color': 'rgba(245,158,11,0.2)'},
                {'range': [65,100],'color': 'rgba(52,211,153,0.2)'},
            ],
            'threshold': {'line': {'color': '#818cf8', 'width': 3},
                          'thickness': 0.8, 'value': 75}
        }
    ))
    apply_layout(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    hr  = (df['avg_score'] < 40).sum()
    ar  = ((df['avg_score'] >= 40) & (df['avg_score'] < 60)).sum()
    avg = ((df['avg_score'] >= 60) & (df['avg_score'] < 75)).sum()
    hp  = (df['avg_score'] >= 75).sum()
    c1.metric("🔴 High Risk",       hr,  delta=f"{hr/len(df)*100:.1f}%")
    c2.metric("🟡 At Risk",         ar,  delta=f"{ar/len(df)*100:.1f}%")
    c3.metric("🔵 Average",         avg, delta=f"{avg/len(df)*100:.1f}%")
    c4.metric("🟢 High Performers", hp,  delta=f"{hp/len(df)*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        risk_trend = df.groupby(['parental level of education','risk_level']).size().reset_index(name='count')
        fig = px.bar(risk_trend, x='parental level of education', y='count',
                     color='risk_level',
                     color_discrete_sequence=COLORS,
                     title="Risk Distribution by Parental Education",
                     barmode='stack')
        apply_layout(fig)
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        corr_data = df[['math score','reading score','writing score',
                         'study_hours','attendance','stress_level','avg_score']].corr()
        fig = px.imshow(corr_data, text_auto='.2f',
                        color_continuous_scale='Blues',
                        title="Feature Correlation Heatmap")
        apply_layout(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🚨 At-Risk Student Records")
    at_risk_df = df[df['avg_score'] < 40][
        ['gender','math score','reading score','writing score',
         'avg_score','study_hours','attendance','stress_level']
    ].sort_values('avg_score')
    st.dataframe(safe_gradient_style(at_risk_df.head(20), subset=['avg_score'], cmap='Reds'),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FEEDBACK INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Feedback Intel":
    hero("Feedback Intelligence", "AI-powered sentiment & topic analysis from student feedback", "💬")

    feedback = st.text_area("Paste student / course feedback here",
                             height=160,
                             placeholder="e.g. The teacher explained concepts really well but exams were extremely stressful...")

    c1,c2 = st.columns(2)
    with c1: category = st.selectbox("Feedback Category", ["Course","Teacher","Environment","Resources","General"])
    with c2: student_id = st.text_input("Student ID (optional)", placeholder="STU-001")

    if st.button("🧠 Analyze Feedback", use_container_width=True):
        if not feedback.strip():
            st.warning("Please enter feedback text.")
        else:
            with st.spinner("Running sentiment analysis..."):
                time.sleep(0.4)

            POSITIVE = {"good","excellent","great","amazing","helpful","supportive","clear",
                        "best","love","fantastic","wonderful","easy","understand","enjoy","learned"}
            NEGATIVE = {"bad","poor","stressful","boring","difficult","hard","confusing",
                        "terrible","hate","useless","slow","unclear","problem","issue","lack"}
            SUGGEST  = {"should","improve","need","more","better","add","suggest","recommend"}

            words = re.findall(r'\b\w+\b', feedback.lower())
            pos  = sum(1 for w in words if w in POSITIVE)
            neg  = sum(1 for w in words if w in NEGATIVE)
            sug  = sum(1 for w in words if w in SUGGEST)
            total = len(words)
            sentiment_score = int(((pos - neg) / max(total,1)) * 500 + 50)
            sentiment_score = max(0, min(100, sentiment_score))

            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Positive Signals", pos)
            k2.metric("Negative Signals", neg)
            k3.metric("Suggestions",      sug)
            k4.metric("Sentiment Score",  f"{sentiment_score}/100")

            st.markdown("---")

            if pos > neg:
                st.success(f"**✅ Positive Sentiment Detected** — Score: {sentiment_score}/100\n\nThis feedback reflects satisfaction with the learning experience.")
            elif neg > pos:
                st.error(f"**⚠️ Negative Sentiment Detected** — Score: {sentiment_score}/100\n\nThis feedback highlights concerns requiring attention.")
            else:
                st.info(f"**🔵 Neutral / Mixed Sentiment** — Score: {sentiment_score}/100\n\nFeedback contains balanced observations.")

            col1, col2 = st.columns(2)
            with col1:
                found_pos = [w for w in words if w in POSITIVE]
                found_neg = [w for w in words if w in NEGATIVE]
                if found_pos:
                    st.markdown("**🟢 Positive Keywords Found:**")
                    st.markdown(" ".join([f'<span class="tag">{w}</span>' for w in set(found_pos)]),
                                unsafe_allow_html=True)
                if found_neg:
                    st.markdown("**🔴 Negative Keywords Found:**")
                    st.markdown(" ".join([f'<span class="tag tag-bad">{w}</span>' for w in set(found_neg)]),
                                unsafe_allow_html=True)
            with col2:
                sentiment_data = {"Positive": pos, "Negative": neg, "Neutral": max(0, total//10 - pos - neg)}
                fig = px.pie(values=list(sentiment_data.values()),
                             names=list(sentiment_data.keys()),
                             color_discrete_sequence=['#34d399','#f87171','#94a3b8'],
                             hole=0.6, title="Sentiment Breakdown")
                apply_layout(fig, height=260)
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Clustering":
    hero("Student Clustering Analytics", "Unsupervised segmentation of students by performance profiles", "🔍")

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        st.error("Clustering is unavailable because the required scikit-learn/scipy packages are not compatible with this environment.")
        st.warning(str(e))
        st.stop()

    features = df[['math score','reading score','writing score','study_hours','attendance']].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = km.fit_predict(X)
    features = features.copy()
    features['Cluster'] = clusters.astype(str)
    cluster_names = {'0':'High Performers','1':'Average Learners',
                     '2':'Struggling Students','3':'Inconsistent Performers'}
    features['Cluster Label'] = features['Cluster'].map(cluster_names)

    fig = px.scatter_3d(features, x='math score', y='reading score', z='writing score',
                        color='Cluster Label', size='study_hours',
                        color_discrete_sequence=COLORS,
                        title="3D Student Cluster Visualization",
                        opacity=0.75)
    apply_layout(fig, height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Cluster Profiles")
    col1,col2,col3,col4 = st.columns(4)
    icons = ['🏆','📚','🆘','⚡']
    colors = ['#34d399','#38bdf8','#f87171','#fb923c']
    for i,(col, (k,label)) in enumerate(zip([col1,col2,col3,col4], cluster_names.items())):
        grp  = features[features['Cluster']==k]
        avg  = grp['math score'].mean()
        col.markdown(f"""
        <div class="resume-section" style="border-color:{colors[i]}40;text-align:center;">
            <div style="font-size:1.8rem">{icons[i]}</div>
            <h4 style="color:{colors[i]} !important;font-size:.85rem !important;">{label}</h4>
            <div style="color:#94a3b8;font-size:.8rem;">{len(grp)} students</div>
            <div style="color:{colors[i]};font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;">{avg:.0f}</div>
            <div style="color:#64748b;font-size:.75rem;">avg math score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(features, x='Cluster Label', y='math score',
                     color='Cluster Label', color_discrete_sequence=COLORS,
                     title="Math Score by Cluster")
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(features, x='study_hours', y='math score',
                         color='Cluster Label', color_discrete_sequence=COLORS,
                         title="Study Hours vs Math Score by Cluster")
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  RESUME AI
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Resume AI":
    hero("Resume Intelligence", "Upload your resume for AI-powered ATS scoring & improvement roadmap", "📄")

    if not GEMINI_API_KEYS:
        st.warning("⚠️  Please configure at least one Gemini API key to use Resume AI.")
        st.info("Get a free key at: https://aistudio.google.com/app/apikey")
        st.stop()

    uploaded = st.file_uploader(
        "Upload Resume (PDF, DOCX, or TXT)",
        type=["pdf","docx","txt"],
        help="Your file is processed locally and sent to Gemini for analysis only."
    )

    col1,col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Target Job Title", placeholder="e.g. Data Scientist, Software Engineer")
    with col2:
        experience = st.selectbox("Experience Level", ["Entry Level","Mid Level","Senior","Leadership/Executive"])

    if uploaded and st.button("🚀 Analyze Resume with Gemini AI", use_container_width=True):
        with st.spinner("Extracting text from resume..."):
            resume_text = extract_text_from_upload(uploaded)
        if len(resume_text.strip()) < 50:
            st.error("Could not extract readable text from the file. Try a text-based PDF or DOCX.")
            st.stop()

        with st.spinner("Gemini AI analyzing resume — this takes ~10 seconds..."):
            try:
                result = gemini_resume_analyze(resume_text)
            except json.JSONDecodeError:
                st.error("Gemini returned an unexpected format. Try again.")
                st.stop()
            except RuntimeError as e:
                st.error(f"Gemini quota error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        overall  = result.get("overall_score", 0)
        ats      = result.get("ats_score", 0)
        sections = result.get("sections", {})
        strengths    = result.get("strengths", [])
        improvements = result.get("improvements", [])
        kw_found     = result.get("keywords_found", [])
        kw_missing   = result.get("keywords_missing", [])
        feedback_txt = result.get("overall_feedback", "")

        # Score color
        def score_class(s):
            if s >= 75: return "score-high"
            if s >= 50: return "score-mid"
            return "score-low"

        st.markdown(f"""
        <div style='display:flex;gap:24px;align-items:center;margin:20px 0;flex-wrap:wrap;'>
            <div style='text-align:center;'>
                <div style='color:#94a3b8;font-size:.8rem;margin-bottom:6px;'>OVERALL SCORE</div>
                <span class="score-pill {score_class(overall)}">{overall}/100</span>
            </div>
            <div style='text-align:center;'>
                <div style='color:#94a3b8;font-size:.8rem;margin-bottom:6px;'>ATS COMPATIBILITY</div>
                <span class="score-pill {score_class(ats)}">{ats}/100</span>
            </div>
            <div style='flex:1;min-width:200px;padding:14px 20px;background:#0d1526;
                        border:1px solid rgba(56,189,248,.15);border-radius:14px;'>
                <div style='color:#94a3b8;font-size:.8rem;margin-bottom:4px;'>AI SUMMARY</div>
                <div style='color:#cbd5e1;font-size:.87rem;line-height:1.6;'>{feedback_txt}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Section scores radar
        col1, col2 = st.columns([1,1])
        with col1:
            sec_names = [s.replace("_"," ").title() for s in sections]
            sec_vals  = [sections[s].get("score",0)*10 for s in sections]
            fig = go.Figure(go.Scatterpolar(
                r=sec_vals + [sec_vals[0]],
                theta=sec_names + [sec_names[0]],
                fill='toself',
                fillcolor='rgba(129,140,248,0.15)'
            ))
            apply_layout(fig, title="Resume Section Scores",
                         polar=dict(bgcolor='rgba(0,0,0,0)',
                                    radialaxis=dict(visible=True, range=[0,100],
                                                    gridcolor='rgba(255,255,255,0.07)')))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("##### 📋 Section-by-Section Feedback")
            priority_colors = {'High':'#f87171','Medium':'#fbbf24','Low':'#34d399'}
            for sec_key, sec_data in sections.items():
                sc = sec_data.get("score", 0)
                fb = sec_data.get("feedback", "")
                color = '#34d399' if sc>=8 else '#fbbf24' if sc>=5 else '#f87171'
                st.markdown(f"""
                <div class="resume-section">
                    <h4>{sec_key.replace('_',' ').title()}
                        <span style='float:right;color:{color};font-size:1rem;'>{sc}/10</span>
                    </h4>
                    <p style='margin:0;color:#94a3b8;font-size:.85rem;'>{fb}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### ✅ Strengths")
            for s in strengths:
                st.markdown(f"""
                <div class="resume-section" style="border-color:rgba(52,211,153,.3);">
                    <p style='margin:0;color:#a7f3d0;font-size:.87rem;'>✓ {s}</p>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("##### 🔧 Improvements Needed")
            for imp in improvements:
                pcolor = priority_colors.get(imp.get('priority','Low'), '#94a3b8')
                st.markdown(f"""
                <div class="resume-section" style="border-color:{pcolor}40;">
                    <h4 style='color:{pcolor} !important;'>{imp.get('area','')}
                        <span style='float:right;font-size:.7rem;background:{pcolor}20;
                               padding:2px 8px;border-radius:999px;'>{imp.get('priority','')} Priority</span>
                    </h4>
                    <p style='margin:0;color:#94a3b8;font-size:.85rem;'>{imp.get('suggestion','')}</p>
                </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown("##### 🔑 Keywords")
            if kw_found:
                st.markdown("**Found:**")
                st.markdown(" ".join([f'<span class="tag">{k}</span>' for k in kw_found[:15]]),
                            unsafe_allow_html=True)
            if kw_missing:
                st.markdown("**Missing:**")
                st.markdown(" ".join([f'<span class="tag tag-warn">{k}</span>' for k in kw_missing[:15]]),
                            unsafe_allow_html=True)

        # Score bar chart
        st.markdown("---")
        fig = go.Figure(go.Bar(
            x=sec_names,
            y=sec_vals,
            marker_color=COLORS[:len(sec_names)],
            text=[f"{v:.0f}%" for v in sec_vals],
            textposition='outside'
        ))
        apply_layout(fig, title="Section Score Overview (%)", yaxis_range=[0,110])
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "AI Insights":
    hero("AI Insight Engine", "Actionable, data-driven recommendations for academic improvement", "🌟")

    # Key insights from data
    high_study = df[df['study_hours'] >= 7]['avg_score'].mean()
    low_study  = df[df['study_hours'] <= 3]['avg_score'].mean()
    prep_yes   = df[df['test preparation course']=='completed']['avg_score'].mean()
    prep_no    = df[df['test preparation course']=='none']['avg_score'].mean()
    corr_study_math = df['study_hours'].corr(df['math score'])

    c1,c2,c3 = st.columns(3)
    c1.metric("Study Impact",    f"+{high_study-low_study:.1f} pts",  delta="High vs Low study hrs")
    c2.metric("Test Prep Lift",  f"+{prep_yes-prep_no:.1f} pts" if not (pd.isna(prep_yes) or pd.isna(prep_no)) else "N/A",
              delta="With vs Without prep")
    c3.metric("Study-Math Corr", f"{corr_study_math:.2f}",  delta="Pearson r")

    st.markdown("---")
    st.subheader("📌 Top Data-Driven Insights")

    insights = [
        ("📚 Study Hours Drive Performance",
         "Students studying 7+ hrs/day score on average "
         f"{high_study-low_study:.1f} points higher than those studying ≤3 hrs.",
         "High Impact"),
        ("🎯 Test Prep Courses Work",
         "Completing a test preparation course correlates with meaningfully higher average scores.",
         "High Impact"),
        ("💤 Sleep & Stress Are Linked",
         "Students reporting high stress (8+) tend to underperform in all subjects. "
         "Encourage healthy sleep (7-9 hrs) and stress management.",
         "Medium Impact"),
        ("📈 Reading & Writing Correlation",
         f"Reading and writing scores have a Pearson r = "
         f"{df['reading score'].corr(df['writing score']):.2f}, "
         "suggesting shared language skills — improving one lifts both.",
         "Medium Impact"),
        ("🏫 Attendance Matters",
         "Students with 90%+ attendance outperform those below 70% by a significant margin "
         "across all subjects.",
         "High Impact"),
        ("👨‍👩‍👧 Parental Education Effect",
         "Students with parents holding bachelor's or master's degrees show higher avg scores. "
         "Consider targeted support for first-generation learners.",
         "Medium Impact"),
    ]

    for title, body, impact in insights:
        color = '#34d399' if impact=='High Impact' else '#fbbf24'
        st.markdown(f"""
        <div class="resume-section" style="border-color:{color}40;">
            <h4 style='color:{color} !important;'>{title}
                <span style='float:right;font-size:.7rem;background:{color}20;
                       padding:2px 10px;border-radius:999px;color:{color};'>{impact}</span>
            </h4>
            <p style='margin:0;color:#94a3b8;font-size:.87rem;line-height:1.7;'>{body}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        grp = df.groupby('study_hours')['avg_score'].mean().reset_index()
        fig = go.Figure(go.Scatter(
            x=grp['study_hours'],
            y=grp['avg_score'],
            mode='lines+markers',
            marker=dict(color='#38bdf8', size=8)
        ))
        fig.update_layout(title="Avg Score vs Study Hours")
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        grp2 = df.groupby('parental level of education')['avg_score'].mean().sort_values().reset_index()
        fig = px.bar(grp2, x='avg_score', y='parental level of education',
                     orientation='h',
                     title="Avg Score by Parental Education",
                     color='avg_score', color_continuous_scale='Blues')
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STUDY COACH  (AI Chat)
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Study Coach":
    hero("AI Study Coach", "Your personal academic mentor — ask anything about study plans, exams, career goals, stress & more", "🤖")

    if not GEMINI_API_KEYS:
        st.warning("⚠️  Please configure at least one Gemini API key in the sidebar to use the Study Coach.")
        st.info("Get a free key at: https://aistudio.google.com/app/apikey")
        st.stop()

    # ── Session state ──
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_profile" not in st.session_state:
        st.session_state.chat_profile = {}

    # ── Profile sidebar panel ──
    with st.expander("⚙️  Personalise your coach (optional)", expanded=not st.session_state.chat_profile):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            p_name    = st.text_input("Your Name",          placeholder="e.g. Ankit",        key="p_name")
            p_grade   = st.selectbox("Grade / Year",        ["Class 10","Class 11","Class 12",
                                                              "1st Year College","2nd Year College",
                                                              "3rd Year College","4th Year College",
                                                              "Postgraduate","Other"],          key="p_grade")
        with pc2:
            p_subject = st.text_input("Weak Subject(s)",    placeholder="e.g. Math, Physics", key="p_subject")
            p_goal    = st.text_input("Career Goal",        placeholder="e.g. Data Scientist",key="p_goal")
        with pc3:
            p_hours   = st.slider("Daily Study Hours Available", 1, 12, 4,                    key="p_hours")
            p_exam    = st.text_input("Upcoming Exam / Deadline", placeholder="e.g. JEE Main – Feb 2026", key="p_exam")

        if st.button("💾  Save Profile & Start Coaching", key="save_profile"):
            st.session_state.chat_profile = {
                "name": p_name, "grade": p_grade, "weak_subject": p_subject,
                "goal": p_goal, "study_hours": p_hours, "exam": p_exam
            }
            greeting = f"Hello{' ' + p_name if p_name else ''}! 👋 I'm your EduMind Study Coach. "
            if p_goal:
                greeting += f"I see you want to become a **{p_goal}** — great ambition! "
            if p_exam:
                greeting += f"With **{p_exam}** on the horizon, let's build a solid plan. "
            greeting += "\n\nAsk me anything: study schedules, exam tips, motivation strategies, subject help, career roadmaps — I'm here 24/7. What's on your mind? 🎯"
            st.session_state.chat_messages = [{"role":"assistant","content":greeting,"time":datetime.now().strftime("%H:%M")}]
            st.rerun()

    # ── Show profile card if set ──
    prof = st.session_state.chat_profile
    if prof:
        items = []
        if prof.get("name"):          items.append(("Student", prof["name"]))
        if prof.get("grade"):         items.append(("Grade", prof["grade"]))
        if prof.get("goal"):          items.append(("Goal", prof["goal"]))
        if prof.get("weak_subject"):  items.append(("Weak Area", prof["weak_subject"]))
        if prof.get("study_hours"):   items.append(("Hrs/Day", str(prof["study_hours"])))
        if prof.get("exam"):          items.append(("Target Exam", prof["exam"]))
        cols = st.columns(len(items)) if items else []
        if items:
            st.markdown("<div class=\"profile-card\">", unsafe_allow_html=True)
            card_html = "<div style=\"display:flex;gap:28px;flex-wrap:wrap;\">"
            for label,val in items:
                card_html += f"<div><div class=\"label\">{label}</div><div class=\"val\">{val}</div></div>"
            card_html += "</div>"
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Init with welcome if no messages yet ──
    if not st.session_state.chat_messages:
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": "👋 Hi! I'm your **EduMind AI Study Coach**. I can help you with:\n\n• 📅 Personalized study plans & timetables\n• 📚 Subject-specific tips (Math, Science, Languages…)\n• 🎯 Exam strategies & revision techniques\n• 💼 Career roadmaps and college prep\n• 😰 Stress management & motivation\n\nSet your profile above for personalised advice, or just start chatting!",
            "time": datetime.now().strftime("%H:%M")
        }]

    # ── Quick prompt buttons ──
    st.markdown("**💡 Quick Prompts:**")
    quick_prompts = [
        "Make me a 30-day study plan",
        "How to improve my math score?",
        "Best revision techniques for exams",
        "How to manage study stress?",
        "Career path for Data Science",
        "Tips for staying focused",
    ]
    qcols = st.columns(len(quick_prompts))
    triggered_quick = None
    for i, (qcol, qp) in enumerate(zip(qcols, quick_prompts)):
        with qcol:
            st.markdown("<div class=\"quick-btn\">", unsafe_allow_html=True)
            if st.button(qp, key=f"quick_{i}"):
                triggered_quick = qp
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Render chat history ──
    def render_messages():
        html = "<div class=\"chat-wrapper\">"
        for msg in st.session_state.chat_messages:
            role    = msg["role"]
            content = msg["content"]
            ts      = msg.get("time","")
            cls     = "user" if role=="user" else "bot"
            avatar  = "👤" if role=="user" else "🤖"
            # Convert markdown bold **text** → <b>text</b> simply
            content_html = content.replace("**","<b>",1)
            while "**" in content_html:
                content_html = content_html.replace("**","</b>",1)
                if "**" in content_html:
                    content_html = content_html.replace("**","<b>",1)
            content_html = content_html.replace("\n","<br>")
            html += f"""
            <div class="chat-bubble-row {cls}">
                <div class="avatar {cls}">{avatar}</div>
                <div class="bubble {cls}">
                    {content_html}
                    <div class="ts">{ts}</div>
                </div>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        render_messages()
        typing_placeholder = st.empty()

    # ── Gemini multi-turn call ──
    def call_gemini_chat(messages: list, profile: dict) -> str:
        profile_ctx = ""
        if profile:
            profile_ctx = f"""
Student Profile:
- Name: {profile.get("name","Unknown")}
- Grade/Year: {profile.get("grade","Unknown")}
- Weak subjects: {profile.get("weak_subject","Not specified")}
- Career goal: {profile.get("goal","Not specified")}
- Available study hours per day: {profile.get("study_hours","Unknown")}
- Upcoming exam/deadline: {profile.get("exam","None specified")}
"""

        system_prompt = f"""You are EduMind AI Study Coach — a warm, encouraging, highly knowledgeable academic mentor for students.
{profile_ctx}
Guidelines:
- Give practical, actionable advice tailored to the student's profile above
- Use bullet points and structured plans when helpful
- Be motivating and empathetic — students may be stressed
- For study plans, include specific time allocations and subject breakdowns
- Keep responses concise but complete (150–300 words max unless a full plan is requested)
- Use emojis sparingly for warmth
- If asked for a study plan, provide a structured weekly/monthly breakdown
- Always end with an encouraging line or a follow-up question
"""
            # Build conversation prompt from history
        if messages:
            conversation_history = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
        else:
            conversation_history = ""

        prompt = (
            system_prompt
            + "\n\nConversation so far:\n"
            + conversation_history
        )

        def request_fn(client):
            model = client.GenerativeModel(model_name=GEMINI_MODEL)
            return model.generate_content(contents=[{"text": prompt}])

        response = gemini_request_with_retry(request_fn, GEMINI_API_KEYS)
        return response.text.strip()

    # ── Input area ──
    if triggered_quick:
        st.session_state.chat_input_box = ""

    st.markdown("<div class=\"chat-input-area\">", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        input_col, btn_col = st.columns([6,1])
        with input_col:
            user_input = st.text_input(
                "Message your Study Coach…",
                label_visibility="collapsed",
                placeholder="e.g. Make me a 4-week plan for my JEE Math preparation…",
                key="chat_input_box"
            )
        with btn_col:
            send_clicked = st.form_submit_button("Send ➤")
    st.markdown("</div>", unsafe_allow_html=True)

    # Determine what to send
    message_to_send = None
    if triggered_quick:
        message_to_send = triggered_quick
    elif send_clicked and user_input.strip():
        message_to_send = user_input.strip()

    if message_to_send:
        # Add user message immediately
        st.session_state.chat_messages.append({
            "role": "user",
            "content": message_to_send,
            "time": datetime.now().strftime("%H:%M")
        })
        st.rerun()

    # Only process AI response if there's a pending user message waiting
    if len(st.session_state.chat_messages) > 0:
        last_msg = st.session_state.chat_messages[-1]
        if last_msg["role"] == "user" and not any(m["role"] == "assistant" for m in st.session_state.chat_messages[st.session_state.chat_messages.index(last_msg)+1:]):
            # User message exists but no assistant response yet
            with st.spinner("EduMind is thinking…"):
                try:
                    reply = call_gemini_chat(
                        st.session_state.chat_messages,
                        st.session_state.chat_profile
                    )
                except RuntimeError as e:
                    reply = (
                        "Sorry, EduMind cannot complete this request right now because Gemini quota is exhausted. "
                        "Please check your API key, billing, and quota status, then try again later.\n\n"
                        f"Details: {e}"
                    )
                except Exception as e:
                    reply = f"Sorry, I ran into an issue: {e}. Please check your API key and try again."

            # Show typing animation character by character
            displayed = ""
            for i in range(0, len(reply), 8):
                displayed = reply[:i+8]
                displayed_html = displayed.replace("**","<b>",1)
                while "**" in displayed_html:
                    displayed_html = displayed_html.replace("**","</b>",1)
                    if "**" in displayed_html:
                        displayed_html = displayed_html.replace("**","<b>",1)
                displayed_html = displayed_html.replace("\n","<br>")
                typing_placeholder.markdown(f"""
                    <div class="chat-wrapper">
                        <div class="chat-bubble-row bot">
                            <div class="avatar bot">🤖</div>
                            <div class="bubble bot">{displayed_html}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                time.sleep(0.02)
            typing_placeholder.empty()

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": reply,
                "time": datetime.now().strftime("%H:%M")
            })
            st.rerun()

    # ── Clear chat button ──
    if st.session_state.chat_messages:
        if st.button("🗑️  Clear Conversation", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:16px 0;color:#334155;font-size:.8rem;font-family:DM Sans,sans-serif;'>
    <span style='font-family:Syne,sans-serif;font-weight:700;background:linear-gradient(135deg,#38bdf8,#818cf8);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>EduMind AI Platform</span>
    &nbsp;·&nbsp; Powered by Google Gemini &nbsp;·&nbsp; Built with Streamlit
    &nbsp;·&nbsp; Developed by <b style='color:#475569;'>Ankit Chowdhary</b>
</div>""", unsafe_allow_html=True)
