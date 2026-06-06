import pandas as pd

print("Loading data...")
postings = pd.read_csv("data/postings.csv")
job_skills = pd.read_csv("data/jobs/job_skills.csv")
salaries = pd.read_csv("data/jobs/salaries.csv")
companies = pd.read_csv("data/companies/companies.csv")
skills = pd.read_csv("data/mappings/skills.csv")

# ── 1. CLEAN POSTINGS ──────────────────────────────────────────
postings = postings[['job_id', 'company_name', 'title', 'location', 
                      'formatted_work_type', 'formatted_experience_level',
                      'remote_allowed', 'normalized_salary', 'listed_time']].copy()

# Bersihkan title jadi kategori role
def categorize_role(title):
    title = str(title).lower()
    if any(x in title for x in ['data analyst', 'analytics']):
        return 'Data Analyst'
    elif any(x in title for x in ['data scientist', 'data science']):
        return 'Data Scientist'
    elif any(x in title for x in ['machine learning', 'ml engineer', 'ai engineer']):
        return 'ML Engineer'
    elif any(x in title for x in ['data engineer', 'etl']):
        return 'Data Engineer'
    elif any(x in title for x in ['software engineer', 'software developer', 'backend', 'frontend', 'fullstack', 'full stack']):
        return 'Software Engineer'
    elif any(x in title for x in ['business intelligence', 'bi developer', 'bi analyst']):
        return 'BI Developer'
    else:
        return 'Other'

postings['role_category'] = postings['title'].apply(categorize_role)

# Ekstrak negara/state dari location
postings['state'] = postings['location'].apply(
    lambda x: str(x).split(',')[-1].strip() if pd.notna(x) else 'Unknown'
)

# Konversi listed_time ke datetime
postings['listed_date'] = pd.to_datetime(postings['listed_time'], unit='ms', errors='coerce')
postings['month'] = postings['listed_date'].dt.to_period('M').astype(str)

# ── 2. MERGE SKILLS ────────────────────────────────────────────
skills_merged = job_skills.merge(skills, on='skill_abr', how='left')
skills_per_job = skills_merged.groupby('job_id')['skill_name'].apply(list).reset_index()
postings = postings.merge(skills_per_job, on='job_id', how='left')

# ── 3. SAVE ────────────────────────────────────────────────────
postings.to_csv("data/jobs_cleaned.csv", index=False)
print(f"✅ Cleaned data saved! Shape: {postings.shape}")
print(postings['role_category'].value_counts())