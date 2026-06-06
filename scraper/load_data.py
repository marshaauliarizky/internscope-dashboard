import pandas as pd

# Load data
postings = pd.read_csv("data/postings.csv")
job_skills = pd.read_csv("data/jobs/job_skills.csv")
salaries = pd.read_csv("data/jobs/salaries.csv")
companies = pd.read_csv("data/companies/companies.csv")
skills = pd.read_csv("data/mappings/skills.csv")

# Cek isi data
print("=== POSTINGS ===")
print(postings.shape)
print(postings.columns.tolist())
print(postings.head(2))

print("\n=== JOB SKILLS ===")
print(job_skills.shape)
print(job_skills.head(2))

print("\n=== SKILLS ===")
print(skills.head(2))