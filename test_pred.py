from train_model import load_or_train_model, prepare_input_features
model,cols=load_or_train_model()
# use the same values from the screenshot the user provided
student={
    'technical_skill_rating': 9,
    'soft_skill_rating': 9,
    'cgpa': 8,
    'academic_year': 3,
    'num_projects': 4,
    'internship_experience': 2,
    'weekly_upskilling_hours': 12,
}
X=prepare_input_features(student,cols)
print(X)
prob = model.predict_proba(X)[0][1]
readiness_score = round(prob * 100, 2)
# replicate override rules from app.py
cgpa_val = float(student.get('cgpa', 0))
tech_val = int(student.get('technical_skill_rating', 0))
soft_val = int(student.get('soft_skill_rating', 0))
projects_val = int(student.get('num_projects', 0))
internships_val = int(student.get('internship_experience', 0))
hours_val = int(student.get('weekly_upskilling_hours', 0))
override = (
    cgpa_val >= 7.0 or
    tech_val >= 8 or
    soft_val >= 8 or
    projects_val >= 3 or
    internships_val >= 1 or
    hours_val >= 10
)
readiness_status = "Career Ready" if (readiness_score >= 50 or override) else "Not Ready"
if override:
    readiness_score = max(readiness_score, 70.0)
print('pred', model.predict(X)[0], 'proba', model.predict_proba(X))
print('readiness_score', readiness_score, 'status', readiness_status, 'override', override)
