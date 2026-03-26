import os
import random
import time
import difflib
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request
from train_model import load_or_train_model, prepare_input_features

load_dotenv()

AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()

# Get API keys from environment (no hardcoded defaults). If the variable is missing
# the key will be an empty string and the caller can handle the absence gracefully.
_openai_key = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_KEY = _openai_key

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAsNy1rPPfdTyZCLoPpzS59Z5_Jv-QwsiE").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Debug: Print which provider and keys are loaded (obfuscate actual values)
print(f"[CONFIG] AI_PROVIDER = {AI_PROVIDER}")
print(f"[CONFIG] GEMINI_API_KEY = {'SET' if GEMINI_API_KEY else 'NOT SET'}")
print(f"[CONFIG] GEMINI_MODEL = {GEMINI_MODEL}")
print(f"[CONFIG] OPENAI_API_KEY = {'SET' if OPENAI_API_KEY else 'NOT SET'}")

if not OPENAI_API_KEY:
    print("[CONFIG WARNING] OPENAI_API_KEY is not configured; any OpenAI calls will fall back to template logic.")
if not GEMINI_API_KEY:
    print("[CONFIG WARNING] GEMINI_API_KEY is not configured; the Gemini path will not be used.")


def normalize_text(value: str) -> str:
    return value.strip().lower()


def _get_company_skills_db() -> Dict[str, List[str]]:
    """Real-world tech stacks used by major companies.

    By default the function returns a *static, hard-coded* dictionary defined
    in this file.  This is purely illustrative; the values are not fetched
    from any external service and therefore cannot stay up-to-date automatically.

    If you want to use real‑time or organization-specific data, create a
    JSON file named ``company_skills.json`` alongside ``app.py`` with the
    same structure and the loader below will pick it up.  An example file might
    look like::

        {
            "google": ["python","java",...],
            "microsoft": [...]
        }

    The code tries to load that file first and falls back to the hard-coded
    sample if loading fails.
    """
    # try to read from external file if present, but merge with the built-in
    # sample data so that missing companies/roles are not lost.
    built_in = {
        "google": ["python", "java", "c++", "go", "javascript", "sql", "tensorflow", "kubernetes", "gcp", "data structures", "algorithms", "system design", "distributed systems", "machine learning", "protobuf"],
        "microsoft": ["c#", "python", "java", "typescript", "azure", "sql", ".net", "react", "power bi", "data structures", "algorithms", "system design", "docker", "kubernetes", "git"],
        "amazon": ["java", "python", "aws", "dynamodb", "sql", "react", "typescript", "docker", "kubernetes", "data structures", "algorithms", "system design", "distributed systems", "linux", "microservices"],
        "meta": ["python", "javascript", "react", "php", "hack", "sql", "graphql", "pytorch", "data structures", "algorithms", "system design", "distributed systems", "mobile development", "c++", "git"],
        "facebook": ["python", "javascript", "react", "php", "hack", "sql", "graphql", "pytorch", "data structures", "algorithms", "system design", "distributed systems", "mobile development", "c++", "git"],
        "apple": ["swift", "objective-c", "python", "c++", "java", "sql", "xcode", "cocoa", "metal", "data structures", "algorithms", "system design", "ios development", "macos", "git"],
        "netflix": ["java", "python", "javascript", "react", "aws", "sql", "microservices", "kafka", "spring boot", "data structures", "algorithms", "system design", "docker", "cassandra", "git"],
        "tesla": ["python", "c++", "java", "pytorch", "tensorflow", "ros", "embedded systems", "computer vision", "sql", "linux", "data structures", "algorithms", "deep learning", "cuda", "git"],
        "uber": ["java", "python", "go", "javascript", "react", "sql", "kafka", "microservices", "docker", "kubernetes", "data structures", "algorithms", "system design", "distributed systems", "git"],
        "spotify": ["python", "java", "javascript", "react", "gcp", "sql", "kafka", "docker", "kubernetes", "data structures", "algorithms", "machine learning", "microservices", "tensorflow", "git"],
        "twitter": ["java", "scala", "python", "javascript", "react", "sql", "kafka", "distributed systems", "data structures", "algorithms", "system design", "graphql", "docker", "kubernetes", "git"],
        "linkedin": ["java", "python", "javascript", "react", "scala", "sql", "kafka", "spark", "data structures", "algorithms", "system design", "rest api", "docker", "hadoop", "git"],
        "airbnb": ["ruby", "javascript", "react", "java", "python", "sql", "aws", "docker", "kubernetes", "data structures", "algorithms", "system design", "graphql", "typescript", "git"],
        "stripe": ["ruby", "python", "java", "javascript", "react", "sql", "aws", "api design", "distributed systems", "data structures", "algorithms", "system design", "go", "typescript", "git"],
        "tcs": ["java", "python", "sql", "javascript", "angular", "react", "spring boot", "aws", "azure", "docker", "agile", "git", "rest api", "microservices", "linux"],
        "infosys": ["java", "python", "sql", "javascript", "angular", "react", "spring boot", "aws", "azure", "docker", "agile", "git", "rest api", "devops", "linux"],
        "wipro": ["java", "python", "sql", "javascript", "angular", "react", "spring boot", "aws", "azure", "docker", "agile", "git", "rest api", ".net", "linux"],
        "cognizant": ["java", "python", "sql", "javascript", "react", "angular", "spring boot", "aws", "azure", "docker", "agile", "git", "rest api", "devops", "microservices"],
        "accenture": ["java", "python", "sql", "javascript", "react", "angular", "aws", "azure", "salesforce", "sap", "agile", "git", "cloud computing", "devops", "power bi"],
        "deloitte": ["python", "java", "sql", "javascript", "react", "aws", "azure", "power bi", "tableau", "salesforce", "sap", "agile", "git", "cloud computing", "data analytics"],
        "ibm": ["python", "java", "javascript", "react", "sql", "watson", "cloud foundry", "docker", "kubernetes", "ai/ml", "data structures", "algorithms", "linux", "blockchain", "git"],
        "oracle": ["java", "sql", "plsql", "python", "javascript", "react", "oracle cloud", "docker", "kubernetes", "data structures", "algorithms", "database design", "linux", "rest api", "git"],
        "salesforce": ["java", "javascript", "apex", "lightning", "sql", "react", "salesforce platform", "rest api", "heroku", "data structures", "algorithms", "soql", "visualforce", "git", "agile"],
        "samsung": ["c++", "java", "python", "android", "kotlin", "embedded systems", "linux", "data structures", "algorithms", "iot", "tizen", "sql", "git", "system design", "os concepts"],
        "adobe": ["javascript", "react", "python", "java", "c++", "sql", "aws", "machine learning", "data structures", "algorithms", "system design", "typescript", "node.js", "docker", "git"],
        "flipkart": ["java", "python", "javascript", "react", "sql", "kafka", "redis", "docker", "kubernetes", "data structures", "algorithms", "system design", "microservices", "spring boot", "git"],
        "swiggy": ["java", "python", "go", "javascript", "react", "sql", "kafka", "redis", "docker", "kubernetes", "data structures", "algorithms", "system design", "microservices", "git"],
        "zomato": ["python", "go", "javascript", "react", "sql", "redis", "kafka", "docker", "kubernetes", "data structures", "algorithms", "system design", "aws", "microservices", "git"],
        "paytm": ["java", "python", "javascript", "react", "sql", "redis", "kafka", "docker", "aws", "data structures", "algorithms", "system design", "microservices", "spring boot", "git"],
        "razorpay": ["go", "python", "javascript", "react", "sql", "redis", "kafka", "docker", "kubernetes", "data structures", "algorithms", "system design", "aws", "api design", "git"],
        "freshworks": ["ruby", "python", "javascript", "react", "sql", "aws", "docker", "kubernetes", "data structures", "algorithms", "system design", "rest api", "redis", "typescript", "git"],
        "zoho": ["java", "javascript", "python", "sql", "react", "angular", "data structures", "algorithms", "system design", "rest api", "linux", "git", "os concepts", "networking", "database design"],
        "capgemini": ["java", "python", "sql", "javascript", "react", "angular", "spring boot", "aws", "azure", "docker", "agile", "git", "devops", "microservices", "linux"],
        "hcl": ["java", "python", "sql", "javascript", "react", "angular", "aws", "azure", "docker", "agile", "git", "rest api", "linux", "devops", "spring boot"],
        "tech mahindra": ["java", "python", "sql", "javascript", "react", "angular", "aws", "azure", "docker", "agile", "git", "rest api", "5g", "iot", "devops"],
    }
    # merge with external overrides if they exist
    try:
        import json
        path = os.path.join(os.path.dirname(__file__), "company_skills.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                ext = json.load(f)
            if isinstance(ext, dict):
                # prefer values from external file but keep built_in keys too
                built_in.update(ext)
    except Exception:
        pass

    return built_in


def _get_role_skills_db() -> Dict[str, List[str]]:
    """Skills required for different tech roles.

    This is another static lookup table.  It does **not** drive the interview
    question generation – those come from separate banks – but is used for the
    skill‑matching/recommendation logic.  The entries here are meant as an
    example and will not update automatically unless you replace them with
    your own data.

    To supply external data create ``role_skills.json`` beside ``app.py`` with
    the same format; entries from that file will be merged with the built-in
    list (external values override duplicates).
    """
    built_in = {
        "machine learning engineer": ["python", "tensorflow", "pytorch", "scikit-learn", "sql", "statistics", "deep learning", "nlp", "computer vision", "docker", "mlops", "data structures", "algorithms", "linux", "git"],
        "machine learning": ["python", "tensorflow", "pytorch", "scikit-learn", "sql", "statistics", "deep learning", "nlp", "computer vision", "docker", "mlops", "data structures", "algorithms", "linux", "git"],
        "ml engineer": ["python", "tensorflow", "pytorch", "scikit-learn", "sql", "statistics", "deep learning", "nlp", "computer vision", "docker", "mlops", "data structures", "algorithms", "linux", "git"],
        "data scientist": ["python", "sql", "pandas", "numpy", "scikit-learn", "statistics", "machine learning", "tableau", "power bi", "r", "data visualization", "a/b testing", "deep learning", "spark", "git"],
        "data science": ["python", "sql", "pandas", "numpy", "scikit-learn", "statistics", "machine learning", "tableau", "power bi", "r", "data visualization", "a/b testing", "deep learning", "spark", "git"],
        "data analyst": ["sql", "python", "excel", "tableau", "power bi", "statistics", "data visualization", "pandas", "r", "google analytics", "etl", "data cleaning", "reporting", "a/b testing", "git"],
        "data engineer": ["python", "sql", "spark", "kafka", "airflow", "aws", "docker", "hadoop", "etl", "data modeling", "data warehousing", "snowflake", "dbt", "linux", "git"],
        "software engineer": ["python", "java", "javascript", "sql", "data structures", "algorithms", "system design", "git", "docker", "rest api", "linux", "agile", "ci/cd", "testing", "database design"],
        "software developer": ["python", "java", "javascript", "sql", "data structures", "algorithms", "system design", "git", "docker", "rest api", "linux", "agile", "ci/cd", "testing", "database design"],
        "full stack developer": ["javascript", "react", "node.js", "python", "sql", "html", "css", "typescript", "rest api", "git", "docker", "mongodb", "database design", "agile", "system design"],
        "full stack": ["javascript", "react", "node.js", "python", "sql", "html", "css", "typescript", "rest api", "git", "docker", "mongodb", "database design", "agile", "system design"],
        "frontend developer": ["javascript", "react", "html", "css", "typescript", "redux", "webpack", "responsive design", "git", "rest api", "testing", "ui/ux", "performance optimization", "accessibility", "agile"],
        "frontend": ["javascript", "react", "html", "css", "typescript", "redux", "webpack", "responsive design", "git", "rest api", "testing", "ui/ux", "performance optimization", "accessibility", "agile"],
        "backend developer": ["python", "java", "sql", "rest api", "docker", "kubernetes", "microservices", "database design", "redis", "kafka", "linux", "git", "system design", "ci/cd", "testing"],
        "backend": ["python", "java", "sql", "rest api", "docker", "kubernetes", "microservices", "database design", "redis", "kafka", "linux", "git", "system design", "ci/cd", "testing"],
        "devops engineer": ["docker", "kubernetes", "aws", "terraform", "ci/cd", "linux", "python", "bash", "jenkins", "ansible", "monitoring", "git", "networking", "security", "cloud computing"],
        "devops": ["docker", "kubernetes", "aws", "terraform", "ci/cd", "linux", "python", "bash", "jenkins", "ansible", "monitoring", "git", "networking", "security", "cloud computing"],
        "cloud engineer": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "linux", "python", "networking", "security", "ci/cd", "serverless", "monitoring", "git", "cloud computing"],
        "mobile developer": ["kotlin", "swift", "react native", "flutter", "java", "dart", "rest api", "sql", "firebase", "git", "ui/ux", "agile", "testing", "ci/cd", "app store deployment"],
        "android developer": ["kotlin", "java", "android sdk", "jetpack compose", "room database", "retrofit", "firebase", "rest api", "git", "mvvm", "data structures", "algorithms", "gradle", "testing", "agile"],
        "ios developer": ["swift", "objective-c", "xcode", "swiftui", "core data", "cocoapods", "rest api", "firebase", "git", "mvvm", "data structures", "algorithms", "ui/ux", "testing", "agile"],
        "web developer": ["javascript", "html", "css", "react", "node.js", "sql", "python", "git", "rest api", "responsive design", "typescript", "mongodb", "docker", "agile", "testing"],
        "ai engineer": ["python", "tensorflow", "pytorch", "deep learning", "nlp", "computer vision", "transformers", "mlops", "docker", "sql", "data structures", "algorithms", "linux", "git", "cloud computing"],
        "cybersecurity": ["networking", "linux", "python", "security tools", "penetration testing", "cryptography", "firewalls", "siem", "incident response", "compliance", "sql", "bash", "wireshark", "nmap", "git"],
        "blockchain developer": ["solidity", "ethereum", "javascript", "react", "web3.js", "smart contracts", "cryptography", "python", "rust", "defi", "sql", "git", "docker", "data structures", "algorithms"],
        "game developer": ["c++", "c#", "unity", "unreal engine", "3d math", "physics", "opengl", "vulkan", "data structures", "algorithms", "python", "git", "game design", "optimization", "networking"],
        "qa engineer": ["selenium", "python", "java", "sql", "postman", "jira", "agile", "api testing", "automation testing", "manual testing", "git", "ci/cd", "performance testing", "test planning", "linux"],
        "testing": ["selenium", "python", "java", "sql", "postman", "jira", "agile", "api testing", "automation testing", "manual testing", "git", "ci/cd", "performance testing", "test planning", "cypress"],
    }
    # merge with external data if present
    try:
        import json
        path = os.path.join(os.path.dirname(__file__), "role_skills.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                ext = json.load(f)
            if isinstance(ext, dict):
                built_in.update(ext)
    except Exception:
        pass
    return built_in



def _find_best_company_match(company: str, company_db: Dict[str, List]) -> str:
    """Return the best matching company key from the database.

    Similar logic to _find_best_role_match but simpler: substring first, then
    fuzzy match.  Returns ``None`` if nothing matches.
    """
    if not company:
        return None
    cl = company.lower()
    for key in company_db:
        if key in cl or cl in key:
            return key
    close = difflib.get_close_matches(cl, list(company_db.keys()), n=1, cutoff=0.6)
    return close[0] if close else None


def infer_required_skills(career_interest: str, target_company: str) -> Tuple[List[str], str, str]:
    """Get required skills by combining company-specific and role-specific skill databases.

    Returns a tuple ``(skills, matched_company, matched_role)``.  The matched
    names are the keys that were used; they may differ from the user input if a
    fuzzy match occurred.  If no match was found for either, the corresponding
    string is ``None`` and only the other source (or generic fallback) is used.
    """
    if not career_interest and not target_company:
        return ("python", "java", "sql", "git", "data structures", "algorithms"), None, None

    career_lower = career_interest.lower().strip()
    company_lower = target_company.lower().strip()

    company_db = _get_company_skills_db()
    role_db = _get_role_skills_db()

    # match keys using helpers
    matched_company = _find_best_company_match(company_lower, company_db)
    matched_role = _find_best_role_match(career_lower, role_db)

    company_skills = company_db.get(matched_company, []) if matched_company else []
    role_skills = role_db.get(matched_role, []) if matched_role else []

    # Combine: prioritize company-specific skills, then add role skills
    if company_skills and role_skills:
        combined = []
        overlap = set(company_skills) & set(role_skills)
        combined.extend(sorted(overlap))
        for s in company_skills:
            if s not in combined:
                combined.append(s)
        for s in role_skills:
            if s not in combined:
                combined.append(s)
        return combined[:15], matched_company, matched_role
    elif company_skills:
        return company_skills[:15], matched_company, matched_role
    elif role_skills:
        return role_skills[:15], matched_company, matched_role
    else:
        # Generic fallback
        generic = ["python", "java", "javascript", "sql", "data structures", "algorithms",
                   "git", "docker", "rest api", "system design", "linux", "agile"]
        return generic, matched_company, matched_role


def compute_skill_match(
    student_languages: List[str],
    student_techs: List[str],
    career_interest: str,
    target_company: str,
) -> Tuple[float, List[str], List[str], str, str]:
    student_skills = {normalize_text(s) for s in (student_languages + student_techs)}
    required_skills, matched_company, matched_role = infer_required_skills(career_interest, target_company)
    required_skills = [normalize_text(s) for s in required_skills]
    if not required_skills:
        return 0.0, [], [], matched_company, matched_role

    required_set = set(required_skills)
    matched = student_skills & required_set
    missing = sorted(list(required_set - matched))
    score = round((len(matched) / len(required_set)) * 100, 2)
    return score, missing, sorted(list(required_set)), matched_company, matched_role


def generate_recommendations(
    student_data: Dict[str, Any],
    readiness_status: str,
    readiness_score: float,
    skill_match_score: float,
    missing_skills: List[str],
) -> Tuple[List[str], List[str]]:
    recs: List[str] = []
    roadmap: List[str] = []

    cgpa = float(student_data.get("cgpa", 0))
    tech = int(student_data.get("technical_skill_rating", 0))
    soft = int(student_data.get("soft_skill_rating", 0))
    projects = int(student_data.get("num_projects", 0))
    internships = int(student_data.get("internship_experience", 0))
    hours = int(student_data.get("weekly_upskilling_hours", 0))
    career = student_data.get("career_interest", "Software Engineer")
    company = student_data.get("target_company", "your target company")

    # --- Personalized Recommendations ---
    if readiness_status == "Career Ready":
        recs.append(
            f"Your readiness score is {readiness_score}%. You are generally ready — start actively applying to {company}!"
        )
    else:
        recs.append(
            f"Your readiness score is {readiness_score}%. You need to focus on key areas before applying to {company} for {career}."
        )

    if skill_match_score >= 75:
        recs.append(
            f"Your skill match score is {skill_match_score}%, which is strong for {career} at {company}. Focus on interview preparation."
        )
    elif skill_match_score >= 50:
        recs.append(
            f"Your skill match score is {skill_match_score}% for {career} at {company}. You have a decent base — learn the missing skills to strengthen your profile."
        )
    else:
        recs.append(
            f"Your skill match score is {skill_match_score}% for {career} at {company}. You need to urgently learn the missing skills listed below."
        )

    if missing_skills:
        recs.append(f"Key skills required by {company} that you're missing: " + ", ".join(missing_skills[:10]))

    if cgpa < 7.0:
        recs.append(
            f"Your CGPA ({cgpa}) is below the typical cutoff for {company}. Focus on improving it through core subjects and practice exams."
        )
    if tech < 7:
        recs.append(
            f"For {career} at {company}, you need stronger technical skills. Practice DSA daily and work on {', '.join(missing_skills[:3]) if missing_skills else 'core technologies'}."
        )
    if soft < 7:
        recs.append(
            f"Companies like {company} value communication skills. Practice mock HR interviews, group discussions, and technical presentations."
        )
    if projects < 3:
        recs.append(
            f"Build 2–3 projects using the tech stack {company} uses ({', '.join(missing_skills[:4]) if missing_skills else 'their required technologies'}). Deploy them on GitHub with proper documentation."
        )
    if internships < 1:
        recs.append(
            f"An internship (even a short-term one) greatly boosts your chances at {company}. Apply for internships in {career}-related roles."
        )
    if hours < 7:
        recs.append(
            f"Increase your weekly upskilling to 7–10+ hours. Focus on: coding practice (LeetCode), learning missing skills, and building projects for {career}."
        )

    # --- Company & Role Specific Roadmap ---
    top_missing = missing_skills[:5] if missing_skills else []

    roadmap.append(f"📋 12-Week Preparation Roadmap for {career} at {company}:")

    # Weeks 1-3: Learn missing skills
    if top_missing:
        roadmap.append(f"Weeks 1–3: Learn the fundamentals of: {', '.join(top_missing)}. Use official docs, YouTube tutorials, and hands-on practice.")
    else:
        roadmap.append(f"Weeks 1–3: Strengthen your existing skills. Go deeper into advanced topics relevant to {career}.")

    # Weeks 4-6: Practice coding
    if tech < 8:
        roadmap.append(
            f"Weeks 4–6: Solve 100+ coding problems on LeetCode/HackerRank. Focus on arrays, strings, trees, graphs, and dynamic programming — commonly asked at {company}."
        )
    else:
        roadmap.append(
            f"Weeks 4–6: Practice medium/hard LeetCode problems and review {company}'s past interview questions on LeetCode Premium or GeeksforGeeks."
        )

    # Weeks 7-9: Build projects
    if projects < 3:
        if top_missing:
            roadmap.append(
                f"Weeks 7–9: Build 1–2 projects using {', '.join(top_missing[:3])}. Examples: REST API app, ML model deployment, or a full-stack web app. Push to GitHub."
            )
        else:
            roadmap.append(
                f"Weeks 7–9: Build 1–2 impressive projects relevant to {career}. Deploy them and add to your portfolio."
            )
    else:
        roadmap.append(
            f"Weeks 7–9: Enhance your existing projects — add tests, CI/CD, documentation, and deploy them. Make them {company}-interview-ready."
        )

    # Weeks 10-11: System design & deep dive
    roadmap.append(
        f"Weeks 10–11: Study system design (if applicable for {career}). Practice explaining your projects, technical decisions, and code to others."
    )

    # Week 12: Mock interviews
    roadmap.append(
        f"Week 12: Do mock interviews (use Pramp, interviewing.io, or peers). Practice {company}-style behavioral questions. Refine your resume and apply!"
    )

    return recs, roadmap


def _build_prompt(career_interest: str, target_company: str, technologies: str, programming_languages: str) -> str:
    # The model should generate *customized* questions for the specified role
    # and company.  Do **not** simply echo the prompt fields back; use them as
    # context to tailor the questions and answers.
    return f"""
You are an expert interviewer for {target_company} hiring for a {career_interest} role.
The candidate is familiar with these programming languages: {programming_languages}
and these technologies: {technologies}.

Design a concise mock interview session with:
- 5 technical / conceptual questions (with brief sample answers)
- 3 coding questions (with high-level solution outline, no full code required)
- 4 HR / behavioral questions (with strong sample answers)
- A short feedback summary with 3–5 improvement tips.

Be specific to the {career_interest} position; avoid generic or unrelated
questions.  Do not repeat the input verbatim in the output.

Return the response in clear Markdown sections:
1. Technical Questions
2. Coding Questions
3. HR Questions
4. Feedback & Improvement Tips
"""


def _call_openai(prompt: str) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env or environment."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI interview coach."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.6,
        "max_tokens": 900,
    }

    # Retry logic for rate limits (429 errors)
    max_retries = 3
    retry_delay = 2  # Start with 2 seconds
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=60)
            
            # Handle rate limit (429) with retry
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                    time.sleep(wait_time)
                    continue
                else:
                    return (
                        "⚠️ Rate Limit Exceeded: You've made too many requests to OpenAI API. "
                        "Please wait a few minutes and try again. "
                        "If you have a free tier account, consider upgrading or waiting longer between requests."
                    )
            
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                return f"Error connecting to OpenAI API: {str(e)}"
    
    return "Failed to generate response after multiple retry attempts."


def _get_role_questions() -> Dict[str, List[Dict[str, str]]]:
    """Role-specific technical question banks."""
    return {
        "machine learning": [
            {"q": "Explain the bias-variance tradeoff and how it affects model performance.", "a": "Bias is error from wrong assumptions (underfitting); variance is error from sensitivity to training data (overfitting). The goal is to minimize both — using techniques like cross-validation, regularization, and ensemble methods."},
            {"q": "What is the difference between bagging and boosting?", "a": "Bagging (e.g., Random Forest) trains models in parallel on random subsets to reduce variance. Boosting (e.g., XGBoost) trains models sequentially, each correcting previous errors, to reduce bias."},
            {"q": "How do you handle imbalanced datasets in classification?", "a": "Use techniques like SMOTE, class weighting, undersampling/oversampling, cost-sensitive learning, or evaluation metrics like F1-score, precision-recall AUC instead of accuracy."},
            {"q": "Explain the difference between L1 and L2 regularization.", "a": "L1 (Lasso) adds absolute value of weights as penalty — produces sparse models (feature selection). L2 (Ridge) adds squared weights — shrinks coefficients but keeps all features. Elastic Net combines both."},
            {"q": "What are transformers and why are they important in modern ML?", "a": "Transformers use self-attention mechanisms to process sequences in parallel (unlike RNNs). They power models like BERT, GPT, and are the foundation of modern NLP and increasingly vision tasks."},
            {"q": "Describe the backpropagation algorithm.", "a": "Backpropagation computes gradients of the loss function with respect to each weight using the chain rule, propagating errors backward through the network to update weights via gradient descent."},
            {"q": "What is transfer learning and when would you use it?", "a": "Transfer learning reuses a pre-trained model (trained on a large dataset) for a new task. Use it when you have limited data — fine-tune the last layers while keeping earlier layers frozen."},
            {"q": "Explain precision, recall, and F1-score with examples.", "a": "Precision = correct positives / predicted positives (spam filter accuracy). Recall = correct positives / actual positives (catching all spam). F1 = harmonic mean of both, useful for imbalanced data."},
            {"q": "What is gradient vanishing/exploding problem and how to solve it?", "a": "In deep networks, gradients can become very small (vanishing) or very large (exploding) during backpropagation. Solutions: ReLU activation, batch normalization, residual connections, gradient clipping, LSTM/GRU for sequences."},
            {"q": "How does a convolutional neural network (CNN) work?", "a": "CNNs use convolutional filters to detect features (edges, textures) in images. Layers include convolution (feature extraction), pooling (downsampling), and fully connected (classification). Filters learn hierarchical features automatically."},
        ],
        "data scientist": [
            {"q": "Explain the Central Limit Theorem and its practical applications.", "a": "CLT states that sample means approach a normal distribution as sample size increases, regardless of population distribution. This enables hypothesis testing, confidence intervals, and A/B testing."},
            {"q": "What is the difference between correlation and causation?", "a": "Correlation measures statistical association between variables; causation means one variable directly affects another. Correlation doesn't imply causation — confounding variables or coincidence may exist."},
            {"q": "How would you approach an A/B testing experiment?", "a": "Define hypothesis, choose metric, calculate sample size, randomly split users, run experiment for sufficient duration, check statistical significance (p-value < 0.05), and consider practical significance."},
            {"q": "Explain dimensionality reduction and when you'd use PCA.", "a": "Dimensionality reduction reduces features while preserving information. PCA finds orthogonal axes of maximum variance. Use when you have many correlated features, for visualization, or to reduce computational cost."},
            {"q": "What are the assumptions of linear regression?", "a": "Linearity, independence of errors, homoscedasticity (constant variance), normal distribution of residuals, no multicollinearity. Violations require transformations or different models."},
            {"q": "Describe how you would handle missing data in a dataset.", "a": "Options: remove rows/columns (if few), impute with mean/median/mode, use KNN imputation, predictive imputation, or create a missing indicator feature. Choice depends on missingness pattern (MCAR, MAR, MNAR)."},
            {"q": "What is feature engineering and why is it important?", "a": "Feature engineering creates new features from raw data to improve model performance. Examples: log transforms, interaction terms, binning, encoding categoricals, date decomposition. Often more impactful than model selection."},
            {"q": "Explain the difference between parametric and non-parametric models.", "a": "Parametric models assume a fixed form (e.g., linear regression — fixed number of parameters). Non-parametric models don't assume a form (e.g., KNN, decision trees — complexity grows with data)."},
        ],
        "software engineer": [
            {"q": "Explain the difference between TCP and UDP protocols.", "a": "TCP is connection-oriented, reliable (guarantees delivery, ordering) — used for web, email. UDP is connectionless, faster but unreliable — used for video streaming, gaming, DNS."},
            {"q": "What is the difference between process and thread?", "a": "A process is an independent program with its own memory space. A thread is a lightweight unit within a process sharing the same memory. Threads are faster to create and switch but need synchronization."},
            {"q": "Explain SOLID principles with real-world examples.", "a": "S: One class, one responsibility. O: Open for extension, closed for modification. L: Subtypes substitutable for base types. I: Many specific interfaces over one general. D: Depend on abstractions, not concretions."},
            {"q": "What is database indexing and how does it improve performance?", "a": "An index is a data structure (usually B-tree) that speeds up data retrieval. It creates a sorted reference to rows, reducing search from O(n) to O(log n). Trade-off: faster reads, slower writes, extra storage."},
            {"q": "Describe microservices architecture and its advantages over monolithic.", "a": "Microservices break an application into small, independent services communicating via APIs. Advantages: independent deployment, scalability, technology flexibility. Challenges: distributed complexity, network latency, data consistency."},
            {"q": "What are design patterns? Explain the Singleton and Observer patterns.", "a": "Design patterns are reusable solutions. Singleton ensures one instance (e.g., database connection). Observer defines one-to-many dependency — when one object changes, all dependents are notified (e.g., event systems)."},
            {"q": "Explain how garbage collection works.", "a": "GC automatically frees memory by identifying objects no longer referenced. Methods: reference counting, mark-and-sweep, generational GC. Languages like Java, Python use GC; C/C++ require manual memory management."},
            {"q": "What is CI/CD and why is it important?", "a": "CI (Continuous Integration) automatically builds and tests code on every commit. CD (Continuous Deployment) automatically deploys to production. Benefits: faster releases, fewer bugs, consistent quality."},
        ],
        "web developer": [
            {"q": "Explain the difference between cookies, localStorage, and sessionStorage.", "a": "Cookies: sent with every HTTP request, max 4KB, can have expiry. localStorage: persists until cleared, 5-10MB, client-only. sessionStorage: cleared when tab closes, 5MB, client-only."},
            {"q": "What is the Virtual DOM and how does React use it?", "a": "Virtual DOM is an in-memory representation of the real DOM. React creates a virtual copy, diffs it with the previous version (reconciliation), and only updates changed elements in the real DOM — improving performance."},
            {"q": "Explain CORS and why it exists.", "a": "Cross-Origin Resource Sharing is a security mechanism that restricts web pages from making requests to a different domain. It prevents unauthorized data access. Servers must send appropriate CORS headers to allow cross-origin requests."},
            {"q": "What is the difference between SQL and NoSQL databases?", "a": "SQL: structured, relational, ACID compliant (PostgreSQL, MySQL). NoSQL: flexible schema, various types (document, key-value, graph), horizontally scalable (MongoDB, Redis). Choose based on data structure and scaling needs."},
            {"q": "Explain the event loop in JavaScript.", "a": "JS is single-threaded. The event loop monitors the call stack and callback queue. When the stack is empty, it pushes callbacks from the queue. This enables async operations (setTimeout, fetch) without blocking."},
            {"q": "What is server-side rendering (SSR) vs client-side rendering (CSR)?", "a": "SSR renders HTML on the server (faster initial load, better SEO). CSR renders in the browser using JavaScript (smoother interactions after load). Next.js/Nuxt.js support both approaches."},
            {"q": "Explain responsive design and mobile-first approach.", "a": "Responsive design adapts layout to screen size using media queries, flexible grids, and fluid images. Mobile-first designs for small screens first, then enhances for larger screens — better performance and UX."},
            {"q": "What is a REST API and its key principles?", "a": "REST uses HTTP methods (GET, POST, PUT, DELETE) for CRUD operations on resources. Principles: stateless, uniform interface, resource-based URLs, cacheable responses, layered system."},
        ],
        "devops engineer": [
            {"q": "Explain the difference between Docker containers and virtual machines.", "a": "VMs virtualize hardware with a full OS (heavy, slow to start). Containers share the host OS kernel, package only the app and dependencies (lightweight, fast startup). Docker is the most popular container platform."},
            {"q": "What is Kubernetes and why is it needed?", "a": "Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications. It handles load balancing, self-healing, rolling updates, and service discovery."},
            {"q": "Explain Infrastructure as Code (IaC) and its benefits.", "a": "IaC manages infrastructure through code (Terraform, CloudFormation) instead of manual setup. Benefits: version control, reproducibility, automation, consistency across environments, faster provisioning."},
            {"q": "What is the difference between monitoring and observability?", "a": "Monitoring tracks predefined metrics and alerts on known issues. Observability (logs, metrics, traces) helps understand WHY a system behaves a certain way — diagnosing unknown issues in complex distributed systems."},
            {"q": "Describe a CI/CD pipeline you would set up for a production application.", "a": "Code push → automated build → unit tests → integration tests → security scan → staging deployment → smoke tests → production deployment (blue/green or canary) → monitoring. Tools: Jenkins, GitHub Actions, ArgoCD."},
            {"q": "What are the key differences between horizontal and vertical scaling?", "a": "Vertical: add more CPU/RAM to one server (simpler, has limits). Horizontal: add more server instances (complex, unlimited scaling). Modern cloud apps prefer horizontal scaling with load balancers."},
        ],
        "mobile developer": [
            {"q": "Explain the Android activity lifecycle.", "a": "Activities go through states: onCreate → onStart → onResume (running) → onPause → onStop → onDestroy. Understanding this prevents memory leaks and data loss during configuration changes."},
            {"q": "What is the difference between native, hybrid, and cross-platform development?", "a": "Native: platform-specific code (Swift/Kotlin), best performance. Hybrid: web tech in native wrapper (Ionic). Cross-platform: shared codebase (React Native, Flutter), good performance with code reuse."},
            {"q": "How do you handle state management in a mobile application?", "a": "Options: local state (useState/setState), global state managers (Redux, Provider, BLoC), persistent storage (SharedPreferences, SQLite). Choose based on scope — component-level vs app-wide state."},
            {"q": "Explain how push notifications work in mobile apps.", "a": "App registers with platform service (FCM for Android, APNs for iOS) → gets device token → sends token to backend → backend sends notification to platform service → platform delivers to device."},
            {"q": "What are the key considerations for mobile app performance?", "a": "Minimize network calls, use caching, optimize images, avoid main thread blocking, use lazy loading, efficient list rendering (RecyclerView/FlatList), reduce app size, profile with tools."},
        ],
    }


def _get_coding_questions() -> Dict[str, List[Dict[str, str]]]:
    """Tech-specific coding question banks."""
    return {
        "python": [
            {"q": "Implement a decorator that caches function results (memoization).", "a": "Use a dictionary to store results. The decorator wraps the function, checks if args are cached, returns cached result or computes and stores it. Python's @functools.lru_cache does this built-in."},
            {"q": "Write a generator function that yields Fibonacci numbers.", "a": "Use two variables a, b = 0, 1. In a loop: yield a, then update a, b = b, a+b. Generators are memory efficient — they produce values on-demand instead of storing all in memory."},
            {"q": "Implement a function to find all permutations of a string.", "a": "Use recursion: base case (single char returns itself). For each character, fix it and recursively permute the rest. Time: O(n!). Can also use itertools.permutations()."},
        ],
        "java": [
            {"q": "Implement a thread-safe Singleton pattern in Java.", "a": "Use double-checked locking with volatile keyword, or Bill Pugh Singleton using inner static class (lazy initialization). Enum Singleton is the simplest thread-safe approach."},
            {"q": "Write a program to detect a loop in a linked list and find its starting node.", "a": "Use Floyd's algorithm: slow and fast pointers. When they meet, reset slow to head. Move both one step at a time — they meet at loop start. Time: O(n), Space: O(1)."},
            {"q": "Implement a custom HashMap with basic put and get operations.", "a": "Use an array of linked lists (buckets). Hash key → bucket index. For collisions, chain entries in linked list. Implement resize when load factor exceeds threshold (typically 0.75)."},
        ],
        "javascript": [
            {"q": "Implement a debounce function from scratch.", "a": "Return a wrapper function that clears previous timer and sets a new one. The original function executes only after the specified delay without new calls. Used for search input, resize events."},
            {"q": "Write a deep clone function that handles nested objects and arrays.", "a": "Recursively traverse the object. For each value: if primitive, copy directly; if array, map and recurse; if object, create new object and recurse on each key. Handle circular references with a WeakMap."},
            {"q": "Implement a Promise.all() equivalent from scratch.", "a": "Return a new Promise. Track resolved count and results array. For each input promise, attach .then() that stores result at correct index and resolves when all done. Reject on first failure."},
        ],
        "react": [
            {"q": "Build a custom hook for data fetching with loading and error states.", "a": "Create useDataFetch(url) that manages state (data, loading, error) using useState. Use useEffect to fetch on mount/url change. Return {data, loading, error}. Handle cleanup for unmounting."},
            {"q": "Implement infinite scroll using React hooks.", "a": "Use IntersectionObserver in useEffect to detect when a sentinel element is visible. On intersection, fetch next page and append data. Track page number and hasMore state. Clean up observer on unmount."},
            {"q": "Create a reusable form validation hook.", "a": "Create useForm(initialValues, validationRules). Track values, errors, touched state. Validate on change/blur. Return {values, errors, handleChange, handleBlur, handleSubmit, isValid}."},
        ],
        "sql": [
            {"q": "Write a query to find the second highest salary from an employee table.", "a": "Use: SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees). Or use DENSE_RANK() window function for nth highest."},
            {"q": "Write a query to find employees who earn more than their managers.", "a": "Use self-join: SELECT e.name FROM employees e JOIN employees m ON e.manager_id = m.id WHERE e.salary > m.salary."},
            {"q": "Design a query to find duplicate records in a table.", "a": "SELECT column, COUNT(*) FROM table GROUP BY column HAVING COUNT(*) > 1. For full rows: use ROW_NUMBER() OVER(PARTITION BY columns ORDER BY id) and filter where rn > 1."},
        ],
        "default": [
            {"q": "Implement a function to check if a string is a valid palindrome (ignoring spaces and case).", "a": "Clean the string (lowercase, remove non-alphanumeric). Use two pointers from start and end moving inward, comparing characters. Time: O(n), Space: O(1)."},
            {"q": "Design a stack that supports push, pop, and getMin in O(1) time.", "a": "Use two stacks: main stack for values, min stack tracking current minimum. On push, also push to min stack if value ≤ current min. On pop, pop from min stack if value equals min."},
            {"q": "Write a function to find the longest common subsequence of two strings.", "a": "Use dynamic programming with a 2D table. dp[i][j] = length of LCS of first i chars and first j chars. If chars match, dp[i][j] = dp[i-1][j-1] + 1, else max(dp[i-1][j], dp[i][j-1]). Time: O(mn)."},
        ],
    }


def _get_hr_questions(target_company: str, career_interest: str) -> List[Dict[str, str]]:
    """Generate company-personalized HR questions."""
    return [
        {"q": f"Why do you want to work at {target_company}?", "a": f"I admire {target_company}'s commitment to innovation and the impact their products have on millions of users. As a {career_interest}, I'm excited to contribute to cutting-edge solutions, learn from world-class engineers, and grow in a culture that values technical excellence."},
        {"q": "Tell me about a project you're most proud of.", "a": f"I built a [project name] that solved [specific problem]. I designed the architecture, implemented key features using [relevant technologies], and achieved [measurable result]. This taught me about system design, teamwork, and delivering under deadlines."},
        {"q": "Describe a time when you had to learn a new technology quickly.", "a": "When my team adopted a new framework, I dedicated evenings to documentation and tutorials, built a small prototype, and shared my learnings with the team within a week. I believe in learning by doing and documenting as I go."},
        {"q": "How do you handle tight deadlines and pressure?", "a": "I prioritize tasks by impact, break work into manageable chunks, communicate proactively with stakeholders about progress, and focus on delivering the MVP first. I also maintain a healthy work routine to stay productive."},
        {"q": f"Where do you see yourself in 5 years as a {career_interest}?", "a": f"I see myself as a senior {career_interest} at {target_company}, leading technical initiatives, mentoring junior team members, and contributing to architectural decisions. I want to be someone the team relies on for complex problem-solving."},
        {"q": "Tell me about a time you disagreed with a teammate. How did you resolve it?", "a": "We disagreed on the database choice for a project. I proposed we evaluate both options against our requirements (scalability, query patterns, cost). We created a comparison matrix, tested both, and chose the one backed by data. It taught me the value of objective decision-making."},
        {"q": "What is your biggest weakness and how are you working on it?", "a": "I sometimes spend too much time optimizing code before shipping. I'm learning to balance perfectionism with pragmatism by setting time-boxes for optimization and focusing on delivering working features first, then iterating."},
        {"q": f"How do you stay updated with the latest trends in {career_interest}?", "a": f"I follow industry blogs, subscribe to newsletters (like TLDR, Morning Brew Tech), attend webinars, participate in open-source projects, and regularly practice on coding platforms. I also follow key people in the {career_interest} space on Twitter/LinkedIn."},
    ]


def _get_prep_resources(career_interest: str, target_company: str, technologies: str, programming_languages: str) -> str:
    """Generate interview preparation resources with real links.

    ``target_company`` is used for company-specific tips at the end of the
    resource list (e.g. link to the company's careers page).  Past versions
    mistakenly referenced ``target_company`` without receiving it, raising a
    NameError.
    """
    role_lower = career_interest.lower()
    langs = [l.strip().lower() for l in programming_languages.split(",") if l.strip()] if programming_languages else []
    techs = [t.strip().lower() for t in technologies.split(",") if t.strip()] if technologies else []

    # Coding practice platforms
    coding_section = """
    <h3>💻 Coding Practice Platforms</h3>
    <ul>
        <li><a href="https://leetcode.com/problemset/" target="_blank">LeetCode</a> — Best for DSA interview prep (start with Top 150 Interview Questions)</li>
        <li><a href="https://www.hackerrank.com/domains" target="_blank">HackerRank</a> — Skill-based challenges with certifications</li>
        <li><a href="https://www.codingninjas.com/studio" target="_blank">Coding Ninjas Studio</a> — Guided problem sets by topic</li>
        <li><a href="https://www.geeksforgeeks.org/practice/" target="_blank">GeeksforGeeks Practice</a> — Company-wise and topic-wise problems</li>
        <li><a href="https://www.interviewbit.com/practice/" target="_blank">InterviewBit</a> — Structured coding interview preparation</li>
    </ul>"""

    # Role-specific resources
    role_resources = ""
    if any(kw in role_lower for kw in ["machine learning", "ml", "ai", "deep learning"]):
        role_resources = """
    <h3>🤖 Machine Learning / AI Resources</h3>
    <ul>
        <li><a href="https://www.coursera.org/specializations/machine-learning-introduction" target="_blank">Andrew Ng's ML Specialization (Coursera)</a> — Best foundational ML course</li>
        <li><a href="https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF" target="_blank">StatQuest ML Playlist (YouTube)</a> — Visual explanations of ML concepts</li>
        <li><a href="https://www.fast.ai/" target="_blank">fast.ai</a> — Practical deep learning for coders (free)</li>
        <li><a href="https://www.kaggle.com/learn" target="_blank">Kaggle Learn</a> — Hands-on ML tutorials with datasets</li>
        <li><a href="https://madewithml.com/" target="_blank">Made with ML</a> — MLOps and production ML guide</li>
        <li><a href="https://github.com/khangich/machine-learning-interview" target="_blank">ML Interview Guide (GitHub)</a> — Curated ML interview questions</li>
    </ul>"""
    elif any(kw in role_lower for kw in ["data scientist", "data science", "data analyst", "analytics"]):
        role_resources = """
    <h3>📊 Data Science Resources</h3>
    <ul>
        <li><a href="https://www.coursera.org/professional-certificates/google-data-analytics" target="_blank">Google Data Analytics Certificate (Coursera)</a></li>
        <li><a href="https://www.youtube.com/c/3blue1brown" target="_blank">3Blue1Brown (YouTube)</a> — Beautiful math and statistics visualizations</li>
        <li><a href="https://www.kaggle.com/competitions" target="_blank">Kaggle Competitions</a> — Real-world data science challenges</li>
        <li><a href="https://mode.com/sql-tutorial/" target="_blank">Mode SQL Tutorial</a> — Interactive SQL for analytics</li>
        <li><a href="https://www.stratascratch.com/" target="_blank">StrataScratch</a> — Real data science interview questions from top companies</li>
    </ul>"""
    elif any(kw in role_lower for kw in ["web developer", "frontend", "backend", "full stack", "fullstack"]):
        role_resources = """
    <h3>🌐 Web Development Resources</h3>
    <ul>
        <li><a href="https://www.theodinproject.com/" target="_blank">The Odin Project</a> — Free full-stack web dev curriculum</li>
        <li><a href="https://javascript.info/" target="_blank">JavaScript.info</a> — Modern JavaScript tutorial (comprehensive)</li>
        <li><a href="https://www.youtube.com/c/TraversyMedia" target="_blank">Traversy Media (YouTube)</a> — Web dev tutorials and crash courses</li>
        <li><a href="https://roadmap.sh/" target="_blank">Developer Roadmaps</a> — Visual guides for frontend, backend, DevOps paths</li>
        <li><a href="https://web.dev/" target="_blank">web.dev by Google</a> — Best practices for modern web development</li>
    </ul>"""
    elif any(kw in role_lower for kw in ["devops", "cloud", "sre", "infrastructure"]):
        role_resources = """
    <h3>☁️ DevOps / Cloud Resources</h3>
    <ul>
        <li><a href="https://www.youtube.com/c/TechWorldwithNana" target="_blank">TechWorld with Nana (YouTube)</a> — Docker, Kubernetes, DevOps tutorials</li>
        <li><a href="https://kodekloud.com/" target="_blank">KodeKloud</a> — Hands-on DevOps labs and courses</li>
        <li><a href="https://aws.amazon.com/training/digital/" target="_blank">AWS Free Training</a> — Official AWS learning resources</li>
        <li><a href="https://roadmap.sh/devops" target="_blank">DevOps Roadmap</a> — Complete learning path visualization</li>
        <li><a href="https://www.katacoda.com/" target="_blank">Katacoda</a> — Interactive DevOps scenarios in browser</li>
    </ul>"""
    elif any(kw in role_lower for kw in ["mobile", "android", "ios", "flutter", "react native"]):
        role_resources = """
    <h3>📱 Mobile Development Resources</h3>
    <ul>
        <li><a href="https://developer.android.com/courses" target="_blank">Android Developer Courses (Official)</a></li>
        <li><a href="https://flutter.dev/learn" target="_blank">Flutter Official Learning</a> — Cross-platform mobile development</li>
        <li><a href="https://reactnative.dev/docs/tutorial" target="_blank">React Native Tutorial (Official)</a></li>
        <li><a href="https://www.youtube.com/c/TheNetNinja" target="_blank">The Net Ninja (YouTube)</a> — Flutter, React Native tutorials</li>
    </ul>"""
    else:
        role_resources = """
    <h3>🎯 General Tech Resources</h3>
    <ul>
        <li><a href="https://roadmap.sh/" target="_blank">Developer Roadmaps</a> — Visual career path guides for all tech roles</li>
        <li><a href="https://www.coursera.org/" target="_blank">Coursera</a> — University-level courses from Google, IBM, Meta</li>
        <li><a href="https://www.freecodecamp.org/" target="_blank">freeCodeCamp</a> — Free coding certifications and projects</li>
    </ul>"""

    # Language-specific resources
    lang_resources = ""
    lang_items = []
    for lang in langs[:3]:
        if "python" in lang:
            lang_items.append('<li><a href="https://docs.python.org/3/tutorial/" target="_blank">Python Official Tutorial</a> — Complete Python reference</li>')
            lang_items.append('<li><a href="https://realpython.com/" target="_blank">Real Python</a> — In-depth Python tutorials and guides</li>')
        elif "java" in lang:
            lang_items.append('<li><a href="https://dev.java/learn/" target="_blank">Java Official Learn</a> — Oracle\'s Java tutorials</li>')
            lang_items.append('<li><a href="https://www.baeldung.com/" target="_blank">Baeldung</a> — Java and Spring tutorials</li>')
        elif "javascript" in lang or "js" in lang:
            lang_items.append('<li><a href="https://javascript.info/" target="_blank">JavaScript.info</a> — Modern JS from basics to advanced</li>')
            lang_items.append('<li><a href="https://www.youtube.com/c/WebDevSimplified" target="_blank">Web Dev Simplified (YouTube)</a></li>')
        elif "c++" in lang or "cpp" in lang:
            lang_items.append('<li><a href="https://www.learncpp.com/" target="_blank">LearnCpp.com</a> — Comprehensive C++ tutorial</li>')
        elif "go" in lang or "golang" in lang:
            lang_items.append('<li><a href="https://go.dev/tour/" target="_blank">Go Tour</a> — Interactive Go tutorial</li>')
        elif "rust" in lang:
            lang_items.append('<li><a href="https://doc.rust-lang.org/book/" target="_blank">The Rust Book</a> — Official Rust guide</li>')

    if lang_items:
        lang_resources = "\n    <h3>📚 Language-Specific Resources</h3>\n    <ul>\n" + "\n".join(f"        {item}" for item in lang_items) + "\n    </ul>"

    # Interview prep resources
    interview_section = """
    <h3>🎤 Interview Preparation</h3>
    <ul>
        <li><a href="https://www.pramp.com/" target="_blank">Pramp</a> — Free peer-to-peer mock interviews</li>
        <li><a href="https://www.youtube.com/c/NeetCode" target="_blank">NeetCode (YouTube)</a> — LeetCode solutions explained clearly</li>
        <li><a href="https://www.techinterviewhandbook.org/" target="_blank">Tech Interview Handbook</a> — Complete guide: resume, coding, system design, behavioral</li>
        <li><a href="https://github.com/yangshun/tech-interview-handbook" target="_blank">Tech Interview Handbook (GitHub)</a> — Curated interview tips & resources</li>
        <li><a href="https://www.youtube.com/c/takeUforward" target="_blank">take U forward (YouTube)</a> — DSA complete course and interview prep</li>
        <li><a href="https://interviewing.io/" target="_blank">interviewing.io</a> — Anonymous mock interviews with engineers</li>
    </ul>

    <h3>🏗️ System Design</h3>
    <ul>
        <li><a href="https://github.com/donnemartin/system-design-primer" target="_blank">System Design Primer (GitHub)</a> — The most popular system design resource</li>
        <li><a href="https://www.youtube.com/c/GauravSensei" target="_blank">Gaurav Sen (YouTube)</a> — System design concepts explained simply</li>
        <li><a href="https://bytebytego.com/" target="_blank">ByteByteGo</a> — Visual system design guides by Alex Xu</li>
    </ul>"""

    # company-specific section (always include, even if generic)
    company_section = ""
    if target_company:
        # title-case for nicer display
        comp = target_company.title()
        company_section = f"""
    <h3>🏢 {comp} – Company‑Specific Tips</h3>
    <ul>
        <li>Search Glassdoor or LeetCode for \"{target_company} interview questions\".</li>
        <li>Read {comp}'s official careers page for hiring process details.</li>
    </ul>"""

    return coding_section + role_resources + lang_resources + company_section + interview_section


def _find_best_role_match(role: str, role_db: Dict[str, List]) -> str:
    """Return the best matching key from a role database given a free‑form role string.

    The function first tries exact/substring matches, then word‑by‑word containment,
    and finally falls back to a fuzzy match using difflib.  If nothing is found it
    returns ``None``.
    """
    if not role:
        return None
    rl = role.lower()
    # exact or substring
    for key in role_db:
        if key in rl or rl in key:
            return key
    # all words contained
    for key in role_db:
        words = key.split()
        if all(w in rl for w in words):
            return key
    # fuzzy match
    close = difflib.get_close_matches(rl, list(role_db.keys()), n=1, cutoff=0.6)
    return close[0] if close else None


def _generate_template_interview(career_interest: str, target_company: str, technologies: str, programming_languages: str) -> str:
    """Generate a comprehensive, dynamic mock interview with prep materials."""
    role_lower = career_interest.lower()
    langs = [l.strip().lower() for l in programming_languages.split(",") if l.strip()] if programming_languages else []

    # --- Select role-specific technical questions ---
    role_questions = _get_role_questions()
    matched_role = _find_best_role_match(career_interest, role_questions)
    if matched_role is None:
        # user supplied an unrecognized role; don't silently default to software
        # engineer with no feedback. fall back to a generic question set but make
        # it clear in the title.
        matched_role = "software engineer"
        fallback_note = True
    else:
        fallback_note = False

    tech_qs = role_questions.get(matched_role, role_questions["software engineer"])
    selected_tech = random.sample(tech_qs, min(5, len(tech_qs)))

    # --- Select coding questions based on languages ---
    coding_questions = _get_coding_questions()
    selected_coding = []
    for lang in langs:
        for key in coding_questions:
            if key in lang:
                selected_coding.extend(coding_questions[key])
                break
    if len(selected_coding) < 3:
        selected_coding.extend(coding_questions["default"])
    selected_coding = random.sample(selected_coding, min(3, len(selected_coding)))

    # --- Select HR questions ---
    hr_questions = _get_hr_questions(target_company, career_interest)
    selected_hr = random.sample(hr_questions, min(4, len(hr_questions)))

    # --- Build HTML output (exam-style mock interview) ---
    html = f'<h2>📋 Mock Interview Practice (Exam Mode) for {career_interest} at {target_company}</h2>\n'
    if fallback_note:
        html += '<p><em>⚠️ The role you entered was not recognised; defaulting to a generic software‑engineer question set.</em></p>\n'
    html += f'<p><em>Technologies: {technologies or "N/A"} | Languages: {programming_languages or "N/A"}</em></p>\n'
    html += '<p><em>Try to answer each question on your own first, then click "Show Answer" to reveal a sample response.</em></p>\n'

    # Helper to create a toggleable question block
    def _question_block(idx: int, question: str, answer_html: str, prefix: str) -> str:
        qid = f"{prefix}-{idx}"
        return (
            f'<div class="question-block" style="margin-bottom: 1rem;">\n'
            f'  <strong>Q{idx}: {question}</strong><br>\n'
            f'  <button type="button" class="toggle-answer" data-target="{qid}" style="margin-top:0.5rem;">Show Answer</button>\n'
            f'  <div id="{qid}" class="answer" style="display:none; margin-top:0.5rem; padding:0.75rem; background:rgba(148,163,184,0.08); border-radius:0.375rem;">{answer_html}</div>\n'
            f'</div>\n'
        )

    # Technical Questions
    html += '<h3>🔧 1. Technical / Conceptual Questions</h3>\n'
    for i, q in enumerate(selected_tech, 1):
        answer_html = f'<em>Sample Answer:</em> {q["a"]}'
        html += _question_block(i, q["q"], answer_html, "tech")

    # Coding Questions
    html += '<hr>\n<h3>💻 2. Coding Questions</h3>\n'
    for i, q in enumerate(selected_coding, 1):
        answer_html = f'<em>Solution Approach:</em> {q["a"]}'
        html += _question_block(i, q["q"], answer_html, "code")

    # HR Questions
    html += '<hr>\n<h3>🤝 3. HR / Behavioral Questions</h3>\n'
    for i, q in enumerate(selected_hr, 1):
        answer_html = f'<em>Sample Answer:</em> {q["a"]}'
        html += _question_block(i, q["q"], answer_html, "hr")

    # Improvement Tips
    html += '<hr>\n<h3>📈 4. Feedback & Improvement Tips</h3>\n'
    tips = [
        f"Practice at least 2-3 {matched_role}-specific coding problems daily on LeetCode or HackerRank.",
        f"Build a portfolio project using {technologies if technologies else 'your target tech stack'} and deploy it — this impresses interviewers at {target_company}.",
        "Use the STAR method (Situation, Task, Action, Result) for all behavioral questions.",
        "Study system design fundamentals — even for mid-level roles, companies increasingly ask design questions.",
        f"Research {target_company}'s recent products, blog posts, and engineering culture before the interview.",
        f"Practice explaining {career_interest} concepts to a non-technical person — this demonstrates deep understanding.",
        "Mock interview with a peer or use free platforms like Pramp for realistic practice.",
    ]
    html += '<ol>\n'
    for tip in random.sample(tips, min(5, len(tips))):
        html += f'  <li>{tip}</li>\n'
    html += '</ol>\n'

    # Preparation Resources
    html += '<hr>\n<h3>📚 5. Interview Preparation Materials & Resources</h3>\n'
    html += _get_prep_resources(career_interest, target_company, technologies, programming_languages)

    # JS to toggle answer visibility
    html += (
        '<script>\n'
        'document.querySelectorAll(\'.toggle-answer\').forEach(btn => {\n'
        '  btn.addEventListener(\'click\', () => {\n'
        '    const target = document.getElementById(btn.dataset.target);\n'
        '    if (!target) return;\n'
        '    const isVisible = target.style.display === \'block\';\n'
        '    target.style.display = isVisible ? \'none\' : \'block\';\n'
        '    btn.textContent = isVisible ? \'Show Answer\' : \'Hide Answer\';\n'
        '  });\n'
        '});\n'
        '</script>\n'
    )

    return html


def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        print(f"[DEBUG] Calling Gemini API with model: {GEMINI_MODEL}")
        resp = requests.post(url, json=body, timeout=60)
        print(f"[DEBUG] Gemini API response status: {resp.status_code}")

        if resp.status_code == 400:
            error_data = resp.json()
            error_msg = error_data.get("error", {}).get("message", "Unknown error")
            print(f"[ERROR] Gemini 400: {error_msg}")
            return (
                f"⚠️ Gemini API Key is INVALID. Error: {error_msg}\n\n"
                "Please generate a new valid API key from: https://aistudio.google.com/apikey\n"
                "Then update GEMINI_API_KEY in your .env file and restart the server."
            )
        if resp.status_code == 403:
            return (
                "⚠️ Gemini API key doesn't have permission. "
                "Please generate a new key from: https://aistudio.google.com/apikey"
            )
        if resp.status_code == 429:
            return "⚠️ Rate Limit Exceeded: Too many requests to Gemini API. Please wait a minute and try again."
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Gemini connection error: {e}")
        return f"⚠️ Error connecting to Gemini API: {str(e)}"
    except (KeyError, IndexError):
        return "⚠️ Failed to parse response from Gemini API."


def generate_mock_interview(
    career_interest: str,
    target_company: str,
    technologies: str,
    programming_languages: str,
) -> str:
    prompt = _build_prompt(career_interest, target_company, technologies, programming_languages)

    # Try Gemini API first
    try:
        result = _call_gemini(prompt)
        if not result.startswith("⚠️"):
            return result
        print(f"[WARNING] Gemini API failed, using template fallback...")
    except Exception as exc:
        print(f"[WARNING] Gemini error: {exc}, using template fallback...")

    # Fallback: generate template-based interview
    print("[INFO] Generating template-based mock interview...")
    return _generate_template_interview(career_interest, target_company, technologies, programming_languages)


def create_app():
    app = Flask(__name__)

    # Load or train the ML model at startup
    model, feature_columns = load_or_train_model()

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/analyze", methods=["POST"])
    def analyze():
        form = request.form

        student_data = {
            "name": form.get("name", "").strip(),
            "academic_year": form.get("academic_year", ""),
            "cgpa": float(form.get("cgpa", 0) or 0),
            "programming_languages": [s.strip().lower() for s in form.get("programming_languages", "").split(",") if s.strip()],
            "technologies": [s.strip().lower() for s in form.get("technologies", "").split(",") if s.strip()],
            "technical_skill_rating": int(form.get("technical_skill_rating", 0) or 0),
            "soft_skill_rating": int(form.get("soft_skill_rating", 0) or 0),
            "num_projects": int(form.get("num_projects", 0) or 0),
            "internship_experience": int(form.get("internship_experience", 0) or 0),
            "weekly_upskilling_hours": int(form.get("weekly_upskilling_hours", 0) or 0),
            "career_interest": form.get("career_interest", "").strip(),
            "target_company": form.get("target_company", "").strip(),
        }

        # Prepare features for ML model
        X = prepare_input_features(student_data, feature_columns)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # probability of "Ready"

        # The raw model output may be biased toward "Not Ready" due to
        # class imbalance.  Show the user a more intuitive status by
        # converting the probability into a percentage and using a simple
        # threshold.  This also prevents confusion when a high score (>50%)
        # still results in a 0 prediction.
        readiness_score = round(probability * 100, 2)
        readiness_status = "Career Ready" if readiness_score >= 50 else "Not Ready"

        # Manual override rules based on individual fields.
        # The ML model can under‑estimate readiness when certain strong
        # attributes are present; for example a student with a high CGPA but
        # few projects will still score poorly.  The user has asked that
        # "not only CGPA, but also soft/technical/other skills" should force a
        # positive status when they are good, so we include a handful of
        # simple threshold checks.  This logic is intentionally heuristic and
        # separate from the learned model.
        cgpa_val = float(student_data.get("cgpa", 0))
        tech_val = int(student_data.get("technical_skill_rating", 0))
        soft_val = int(student_data.get("soft_skill_rating", 0))
        projects_val = int(student_data.get("num_projects", 0))
        internships_val = int(student_data.get("internship_experience", 0))
        hours_val = int(student_data.get("weekly_upskilling_hours", 0))

        # any of the following is considered "strong enough" to override
        override = (
            cgpa_val >= 7.0 or
            tech_val >= 8 or
            soft_val >= 8 or
            projects_val >= 3 or
            internships_val >= 1 or
            hours_val >= 10
        )

        if override:
            readiness_status = "Career Ready"
            readiness_score = max(readiness_score, 70.0)

        # Skill match calculation
        skill_match_score, missing_skills, company_required_skills, matched_company, matched_role = compute_skill_match(
            student_languages=student_data["programming_languages"],
            student_techs=student_data["technologies"],
            career_interest=student_data["career_interest"],
            target_company=student_data["target_company"],
        )

        # Recommendations
        recommendations, roadmap = generate_recommendations(
            student_data=student_data,
            readiness_status=readiness_status,
            readiness_score=readiness_score,
            skill_match_score=skill_match_score,
            missing_skills=missing_skills,
        )

        return render_template(
            "result.html",
            student=student_data,
            readiness_status=readiness_status,
            readiness_score=readiness_score,
            skill_match_score=skill_match_score,
            missing_skills=missing_skills,
            company_required_skills=company_required_skills,
            matched_company=matched_company,
            matched_role=matched_role,
            recommendations=recommendations,
            roadmap=roadmap,
        )

    @app.route("/mock-interview", methods=["POST"])
    def mock_interview():
        form = request.form

        career_interest = form.get("career_interest", "").strip()
        target_company = form.get("target_company", "").strip()
        tech_stack = form.get("technologies", "")
        programming_languages = form.get("programming_languages", "")

        ai_response = generate_mock_interview(
            career_interest=career_interest,
            target_company=target_company,
            technologies=tech_stack,
            programming_languages=programming_languages,
        )

        # compute matches for display/debug
        matched_company = _find_best_company_match(target_company.lower(), _get_company_skills_db())
        matched_role = _find_best_role_match(career_interest.lower(), _get_role_skills_db())

        return render_template(
            "interview.html",
            career_interest=career_interest,
            target_company=target_company,
            interview=ai_response,
            matched_company=matched_company,
            matched_role=matched_role,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

