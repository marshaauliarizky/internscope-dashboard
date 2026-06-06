import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ── CONFIG ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="InternScope — Job Market Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── PALETTE ────────────────────────────────────────────────────
C1  = "#c17c74"   # dusty rose (primary)
C2  = "#d4a5a0"   # rose muted
C3  = "#e8c9c5"   # blush light
C4  = "#f5e6e4"   # blush pale
C5  = "#8b4f4a"   # rose dark
C6  = "#f9f0ef"   # background
C7  = "#6b3530"   # deep rose
SEQ = [C1, C2, C3, "#b8706a", "#e0b8b4", C5, C7, "#dba8a3"]

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"], * {{ font-family: 'Sora', sans-serif !important; }}
  .main {{ background: {C6}; }}
  .block-container {{ padding: 0 2.5rem 2rem 2.5rem; max-width: 1400px; }}

  /* ── TOPBAR ── */
  .topbar {{
    background: white;
    border-bottom: 1px solid #eddad8;
    padding: 1rem 2.5rem;
    margin: 0 -2.5rem 2rem -2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  .topbar-brand {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {C5};
    letter-spacing: 2px;
    text-transform: uppercase;
  }}
  .topbar-sub {{
    font-size: 0.72rem;
    color: #b89e9c;
    letter-spacing: 1px;
    text-transform: uppercase;
  }}

  /* ── HERO ── */
  .hero {{
    background: linear-gradient(135deg, {C5} 0%, {C1} 60%, {C3} 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '◈';
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    opacity: 0.08;
    color: white;
  }}
  .hero-title {{
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
  }}
  .hero-sub {{
    font-size: 0.9rem;
    opacity: 0.8;
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.5px;
  }}
  .hero-tag {{
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 100px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    color: white;
  }}

  /* ── METRICS ── */
  .metric-row {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
  .metric-card {{
    flex: 1;
    background: white;
    border: 1px solid #eddad8;
    border-radius: 14px;
    padding: 1.3rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }}
  .metric-card::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {C1}, {C3});
  }}
  .metric-val {{ font-size: 1.8rem; font-weight: 700; color: {C1}; line-height: 1; }}
  .metric-lbl {{ font-size: 0.65rem; color: #b89e9c; margin-top: 0.35rem;
                 text-transform: uppercase; letter-spacing: 1px; font-weight: 500; }}

  /* ── SECTION ── */
  .sec-label {{
    font-size: 0.65rem;
    font-weight: 600;
    color: {C2};
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.4rem;
  }}
  .sec-title {{
    font-size: 1rem;
    font-weight: 600;
    color: #3d2220;
    margin-bottom: 1rem;
  }}
  .card {{
    background: white;
    border: 1px solid #eddad8;
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
  }}

  /* ── ML CARDS ── */
  .ml-card {{
    background: white;
    border: 1px solid #eddad8;
    border-left: 4px solid {C1};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
  }}
  .ml-skill {{ font-weight: 600; color: {C5}; font-size: 0.9rem; }}
  .ml-bar-bg {{ height: 4px; background: {C4}; border-radius: 10px; margin-top: 0.4rem; }}
  .ml-bar {{ height: 4px; background: {C1}; border-radius: 10px; }}
  .ml-pct {{ font-size: 0.72rem; color: {C2}; margin-top: 0.25rem; }}

  /* ── TABS ── */
  div[data-testid="stTabs"] button {{
    font-family: 'Sora', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #b89e9c !important;
  }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C1} !important;
    border-bottom-color: {C1} !important;
  }}

  /* ── FILTER ROW ── */
  .filter-row {{
    background: white;
    border: 1px solid #eddad8;
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }}
  .filter-label {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #b89e9c;
    font-weight: 600;
    white-space: nowrap;
  }}

  /* HIDE SIDEBAR TOGGLE & BRANDING */
  #MainMenu {{visibility: hidden;}}
  footer {{visibility: hidden;}}
  [data-testid="collapsedControl"] {{display: none;}}
  section[data-testid="stSidebar"] {{display: none;}}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/jobs_cleaned.csv")
    df['skill_name'] = df['skill_name'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else []
    )
    return df

df = load_data()

# ── TOPBAR ─────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div>
    <div class="topbar-brand">◈ InternScope</div>
    <div class="topbar-sub">Job Market Analytics</div>
  </div>
  <div class="topbar-sub">LinkedIn Job Postings &nbsp;·&nbsp; 123,849 records &nbsp;·&nbsp; Kaggle</div>
</div>
""", unsafe_allow_html=True)

# ── FILTERS (horizontal) ───────────────────────────────────────
st.markdown('<div class="filter-label">— Filters</div>', unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns(3)
with fc1:
    roles = ['All Roles'] + sorted([r for r in df['role_category'].unique() if r != 'Other'])
    selected_role = st.selectbox("Role", roles, label_visibility="collapsed")
with fc2:
    work_types = ['All Work Types'] + sorted(df['formatted_work_type'].dropna().unique().tolist())
    selected_work = st.selectbox("Work Type", work_types, label_visibility="collapsed")
with fc3:
    exp_levels = ['All Experience Levels'] + sorted(df['formatted_experience_level'].dropna().unique().tolist())
    selected_exp = st.selectbox("Experience Level", exp_levels, label_visibility="collapsed")

# ── FILTER DATA ────────────────────────────────────────────────
filtered = df[df['role_category'] != 'Other'].copy()
if selected_role != 'All Roles':
    filtered = filtered[filtered['role_category'] == selected_role]
if selected_work != 'All Work Types':
    filtered = filtered[filtered['formatted_work_type'] == selected_work]
if selected_exp != 'All Experience Levels':
    filtered = filtered[filtered['formatted_experience_level'] == selected_exp]

all_skills = [s for sub in filtered['skill_name'] for s in sub]

# ── HERO ───────────────────────────────────────────────────────
role_label = selected_role if selected_role != 'All Roles' else 'Tech Roles'
st.markdown(f"""
<div class="hero">
  <div class="hero-tag">Job Market Intelligence</div>
  <div class="hero-title">InternScope</div>
  <p class="hero-sub">Discover what skills employers are looking for in {role_label} — powered by 123,849 LinkedIn job postings.</p>
</div>
""", unsafe_allow_html=True)

# ── METRICS ────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
for col, val, lbl in zip([m1,m2,m3,m4],
    [f"{len(filtered):,}", f"{filtered['company_name'].nunique():,}",
     f"{int(filtered['remote_allowed'].sum()):,}", f"${filtered['normalized_salary'].median():,.0f}"],
    ["Job Postings", "Companies Hiring", "Remote Positions", "Median Salary (USD)"]):
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Skills Analysis", "Salary & Experience", "ML Insight"
])

LAYOUT = dict(
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(family='Sora', size=11, color='#3d2220'),
    margin=dict(l=0, r=0, t=15, b=0),
)

# ════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-label">Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Jobs by Role</div>', unsafe_allow_html=True)
        rc = df[df['role_category']!='Other']['role_category'].value_counts().reset_index()
        rc.columns = ['Role','Count']
        fig = px.bar(rc, x='Count', y='Role', orientation='h',
                     color='Role', color_discrete_sequence=SEQ, template='plotly_white')
        fig.update_layout(**LAYOUT, height=300, showlegend=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-label">Geography</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Top 10 Locations</div>', unsafe_allow_html=True)
        lc = filtered['state'].value_counts().head(10).reset_index()
        lc.columns = ['Location','Count']
        fig3 = px.pie(lc, values='Count', names='Location',
                      color_discrete_sequence=SEQ, template='plotly_white', hole=0.48)
        fig3.update_layout(**LAYOUT, height=300, legend=dict(font=dict(size=10)))
        fig3.update_traces(textinfo='percent', textfont_size=10)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="sec-label">Work Arrangement</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Work Type Distribution</div>', unsafe_allow_html=True)
        wt = filtered['formatted_work_type'].value_counts().reset_index()
        wt.columns = ['Type','Count']
        fig6 = px.bar(wt, x='Type', y='Count', color='Type',
                      color_discrete_sequence=SEQ, template='plotly_white')
        fig6.update_layout(**LAYOUT, height=260, showlegend=False)
        fig6.update_traces(marker_line_width=0)
        st.plotly_chart(fig6, use_container_width=True)

    with col4:
        st.markdown('<div class="sec-label">Work Location</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Remote Availability</div>', unsafe_allow_html=True)
        rc2 = filtered['remote_allowed'].value_counts().reset_index()
        rc2.columns = ['Remote','Count']
        rc2['Remote'] = rc2['Remote'].map({1.0: 'Remote Allowed'})
        rc2['Remote'] = rc2['Remote'].fillna('Not Specified')
        fig7 = px.pie(rc2, values='Count', names='Remote',
                      color_discrete_sequence=[C1, C4],
                      template='plotly_white', hole=0.5)
        fig7.update_layout(**LAYOUT, height=260)
        st.plotly_chart(fig7, use_container_width=True)

# ════════════════════════════════════════
# TAB 2 — SKILLS ANALYSIS
# ════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="sec-label">Demand</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Top 15 Most In-Demand Skills</div>', unsafe_allow_html=True)
        sc = pd.Series(all_skills).value_counts().head(15).reset_index()
        sc.columns = ['Skill','Count']
        fig2 = px.bar(sc, x='Skill', y='Count',
                      color='Count', color_continuous_scale=[C4, C1, C5],
                      template='plotly_white')
        fig2.update_layout(**LAYOUT, height=320, coloraxis_showscale=False,
                           xaxis_tickangle=-35)
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-label">Cross-role</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Skills × Role Heatmap</div>', unsafe_allow_html=True)
        roles_h = [r for r in df['role_category'].unique() if r != 'Other']
        top10s = pd.Series(all_skills).value_counts().head(10).index.tolist()
        heat = []
        for role in roles_h:
            rs = [s for sub in df[df['role_category']==role]['skill_name'] for s in sub]
            rsc = pd.Series(rs).value_counts()
            for sk in top10s:
                heat.append({'Role':role,'Skill':sk,'Count':rsc.get(sk,0)})
        hp = pd.DataFrame(heat).pivot(index='Role',columns='Skill',values='Count').fillna(0)
        fh = px.imshow(hp, color_continuous_scale=[C4,C1,C5],
                       template='plotly_white', aspect='auto')
        fh.update_layout(**LAYOUT, height=320, xaxis_tickangle=-35)
        st.plotly_chart(fh, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Visual</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Skills Word Cloud</div>', unsafe_allow_html=True)
    if all_skills:
        wc = WordCloud(width=1100, height=280, background_color='white',
                       colormap='RdPu', max_words=80,
                       prefer_horizontal=0.7).generate(' '.join(all_skills))
        fwc, ax = plt.subplots(figsize=(11, 2.8))
        fwc.patch.set_facecolor('white')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fwc)

# ════════════════════════════════════════
# TAB 3 — SALARY & EXPERIENCE
# ════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    sal_df = filtered[filtered['normalized_salary'].notna() &
                      (filtered['normalized_salary'] > 0) &
                      (filtered['normalized_salary'] < 500000)]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-label">Compensation</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Salary Distribution by Role</div>', unsafe_allow_html=True)
        if len(sal_df) > 0:
            fig4 = px.box(sal_df, x='role_category', y='normalized_salary',
                          color='role_category', template='plotly_white',
                          color_discrete_sequence=SEQ)
            fig4.update_layout(**LAYOUT, height=320, showlegend=False,
                               xaxis_title='', yaxis_title='Salary (USD)')
            st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-label">Career Level</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Jobs by Experience Level</div>', unsafe_allow_html=True)
        exp_df = filtered[filtered['formatted_experience_level'].notna()]
        ec = exp_df['formatted_experience_level'].value_counts().reset_index()
        ec.columns = ['Level','Count']
        fig5 = px.bar(ec, x='Level', y='Count', color='Level',
                      template='plotly_white', color_discrete_sequence=SEQ)
        fig5.update_layout(**LAYOUT, height=320, showlegend=False)
        fig5.update_traces(marker_line_width=0)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Median Salary by Role</div>', unsafe_allow_html=True)
    if len(sal_df) > 0:
        sr = sal_df.groupby('role_category')['normalized_salary'].median().reset_index()
        sr.columns = ['Role','Median Salary']
        sr = sr.sort_values('Median Salary', ascending=False)
        fig8 = px.bar(sr, x='Role', y='Median Salary',
                      color='Median Salary',
                      color_continuous_scale=[C4, C1, C5],
                      template='plotly_white', text='Median Salary')
        fig8.update_traces(texttemplate='$%{text:,.0f}', textposition='outside',
                           marker_line_width=0)
        fig8.update_layout(**{**LAYOUT, 'margin': dict(l=0,r=0,t=40,b=0)}, 
                   height=320, coloraxis_showscale=False)
        st.plotly_chart(fig8, use_container_width=True)

# ════════════════════════════════════════
# TAB 4 — ML INSIGHT
# ════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Skill Importance Predictor</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#b89e9c'>A Random Forest classifier trained on 123,849 job postings predicts which skills have the highest salary impact per role.</small>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    @st.cache_data
    def train_model(df):
        dfs = df[df['role_category']!='Other'].copy()
        dfs = dfs[dfs['skill_name'].apply(len) > 0]
        de = dfs.explode('skill_name').dropna(subset=['skill_name'])
        ts = de['skill_name'].value_counts().head(30).index.tolist()
        de = de[de['skill_name'].isin(ts)]
        les, ler = LabelEncoder(), LabelEncoder()
        de['se'] = les.fit_transform(de['skill_name'])
        de['re'] = ler.fit_transform(de['role_category'])
        X = de[['se','re']].values
        y = (de['normalized_salary'].fillna(0) > de['normalized_salary'].median()).astype(int)
        
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, f1_score
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Random Forest':       RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
            'Gradient Boosting':   GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=3),
            'Logistic Regression': LogisticRegression(max_iter=200, random_state=42, class_weight='balanced'),
        }
        
        results = []
        trained = {}
        from sklearn.utils.class_weight import compute_sample_weight
        for name, model in models.items():
            sw = compute_sample_weight('balanced', y_train)
            if name == 'Gradient Boosting':
                model.fit(X_train, y_train, sample_weight=sw)
            else:
                model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results.append({
                'Model': name,
                'Accuracy': round(accuracy_score(y_test, preds) * 100, 2),
                'F1 Score': round(f1_score(y_test, preds) * 100, 2),
            })
            trained[name] = model
        
        best_model = trained['Random Forest']
        return best_model, les, ler, ts, pd.DataFrame(results)

    rf_model, le_skill, le_role, top_skills, model_results = train_model(df)

    # ── MODEL COMPARISON ──────────────────────────────────────
    st.markdown('<div class="sec-label">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Accuracy & F1 Score — All Models</div>', unsafe_allow_html=True)

    mr_melted = model_results.melt(id_vars='Model', value_vars=['Accuracy', 'F1 Score'],
                                   var_name='Metric', value_name='Score')
    fig_cmp = px.bar(mr_melted, x='Model', y='Score', color='Metric',
                     barmode='group', template='plotly_white',
                     color_discrete_sequence=[C1, C3],
                     text='Score')
    fig_cmp.update_traces(texttemplate='%{text}%', textposition='outside', marker_line_width=0)
    fig_cmp.update_layout(**{**LAYOUT, 'margin': dict(l=0, r=0, t=50, b=0)},
                          height=320,
                          yaxis=dict(range=[0, 115], title='Score (%)'),
                          legend=dict(orientation='h', yanchor='top', y=1.12,
                                      xanchor='center', x=0.5))
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Model comparison table
    st.dataframe(
        model_results.style.format({'Accuracy': '{:.2f}%', 'F1 Score': '{:.2f}%'})
                     .highlight_max(subset=['Accuracy','F1 Score'], color='#f5e6e4'),
        use_container_width=True, hide_index=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

    mc1, mc2 = st.columns([1, 2])

    with mc1:
        st.markdown("**Select a role to analyze:**")
        ml_role = st.selectbox("Role", [r for r in df['role_category'].unique() if r != 'Other'], key='ml_role')
        results = []
        for skill in top_skills:
            if skill in le_skill.classes_ and ml_role in le_role.classes_:
                se = le_skill.transform([skill])[0]
                re = le_role.transform([ml_role])[0]
                p = rf_model.predict_proba([[se, re]])[0][1]
                results.append({'skill': skill, 'importance': p})
        rdf = pd.DataFrame(results).sort_values('importance', ascending=False).head(8)

        fig_ml = px.bar(rdf, x='importance', y='skill', orientation='h',
                        color='importance', color_continuous_scale=[C4, C1, C5],
                        template='plotly_white')
        fig_ml.update_layout(**LAYOUT, height=320, coloraxis_showscale=False,
                             yaxis_title='', xaxis_title='Salary Impact Score')
        fig_ml.update_traces(marker_line_width=0)
        st.plotly_chart(fig_ml, use_container_width=True)

    with mc2:
        st.markdown(f"**Top skills for {ml_role} — ranked by predicted salary impact**")
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in rdf.iterrows():
            pct = int(row['importance'] * 100)
            st.markdown(f"""
            <div class="ml-card">
                <div class="ml-skill">{row['skill']}</div>
                <div class="ml-bar-bg">
                    <div class="ml-bar" style="width:{max(pct,5)}%"></div>
                </div>
                <div class="ml-pct">{pct}% salary impact score</div>
            </div>
            """, unsafe_allow_html=True)
