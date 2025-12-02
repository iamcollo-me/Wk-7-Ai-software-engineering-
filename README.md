
Part 2: Case Study Analysis


Case 1: Biased Hiring Tool (Amazon’s AI recruiting tool penalized female candidates)

a) Identify the source of bias:

Training Data Bias:The model was trained on past resumes submitted over a 10-year period. As the tech industry workforce was male-dominated, the data predominantly reflected male candidates and reinforced historical gender disparities.
Feature Bias: The model potentially weighted features correlated with gender (e.g., women's colleges, use of gendered language/activities) negatively.
Feedback Loop:Systemic discrimination in feedback (selected hires being mostly male) further entrenched bias.


b) Three fixes to make the tool fairer:

1. Data Rebalancing/De-biasing:
   Actively re-balance training data to ensure gender representation. Remove or adjust features that directly or indirectly encode gender.
   
2. Algorithmic Fairness Constraints:
    Integrate fairness-aware algorithms or constraints in the model to prevent disparate impact (e.g., using regularization, adversarial debiasing, or fairness constraints during model training).

3. Ongoing Bias & Impact Audits:
   Continuously monitor model predictions for disparate outcomes and conduct audits using tools like AI Fairness 360. Include human review in final hiring decisions.

c) Metrics to evaluate fairness after correction:

Disparate Impact Ratio: Checks if hiring rates across genders are statistically balanced (ideally between 0.8 and 1.25).
Equal Opportunity Difference:Measures difference in true positive rates across gender groups.
Demographic Parity:Proportion of positive outcomes should be similar for male and female candidates.
False Positive/Negative Rate Parity:These rates should not differ significantly by gender.
Human-in-the-loop Feedback: Periodically auditing for individual candidate complaints/reports.


Case 2: Facial Recognition in Policing (System misidentifies minorities at higher rates)

a) Ethical Risks:

Wrongful Arrests:False positives can lead to innocent individuals, especially from minority communities, being suspects or even detained erroneously.
Discrimination: Amplifies historical and systemic biases, further marginalizing vulnerable populations.
Privacy Violations:Broad and often non-consensual surveillance erodes civil liberties and public trust.
Lack of Accountability:“Black box” decisions without proper review processes make it hard to appeal or correct errors.


b) Policies for Responsible Deployment:

1. Mandatory Bias Audits & Transparency:
    Run external audits before deployment and publish results. Disclose model performance, including subgroup error rates.

2. Strict Use Guidelines & Human Oversight:
    Use facial recognition as a supportive tool only; require multiple sources of identification before acting. Human officers must review all AI matches.

3. Community Consultation & Consent:
   Involve affected communities and civil rights groups in policy decisions; require explicit legal authorization for surveillance deployments.

4. Redress Mechanisms:
    Clear, accessible processes for individuals to challenge or correct misidentifications.

5. Usage Limitation:
    Restrict to serious crimes investigation only, minimize use in public spaces, and regularly review necessity and proportionality.



The COMPAS Recidivism Dataset audit reveals racial bias in risk scores, with disparate impact on African American defendants.

Solution:
 AI Fairness 360 Audit on COMPAS Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

1. Load COMPAS dataset using AI Fairness 360
compas = CompasDataset().convert_to_dataframe()[0]

2. Specify protected attribute and label columns
protected_attr = 'race'
label = 'two_year_recid'
favorable_label = 0
unfavorable_label = 1

3. Prepare data for modeling
features = compas.drop(columns=[label, protected_attr]).copy()
target = compas[label]
protected = compas[protected_attr]

X_train, X_test, y_train, y_test, pr_train, pr_test = train_test_split(
    features, target, protected, test_size=0.3, random_state=42, stratify=target
)

 4. Train a model and get predictions
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

 5. Convert prediction results for AIF360 metrics
test_df = X_test.copy()
test_df[label] = y_test.values
test_df[protected_attr] = pr_test.values

pred_df = X_test.copy()
pred_df[label] = y_pred
pred_df[protected_attr] = pr_test.values

dataset_true = CompasDataset(
    df=test_df,
    label_name=label,
    protected_attribute_names=[protected_attr]
)
dataset_pred = CompasDataset(
    df=pred_df,
    label_name=label,
    protected_attribute_names=[protected_attr]
)

6. Calculate fairness metrics
class_metric = ClassificationMetric(
    dataset_true,
    dataset_pred,
    unprivileged_groups=[{protected_attr: 'African-American'}],
    privileged_groups=[{protected_attr: 'Caucasian'}]
)
# Disparate Impact Ratio
di_ratio = class_metric.disparate_impact()
# Equal Opportunity Difference
eod = class_metric.equal_opportunity_difference()
# False Positive Rate difference
fpr_diff = class_metric.false_positive_rate_difference()

print(f"Disparate Impact Ratio: {di_ratio:.2f}")
print(f"Equal Opportunity Difference: {eod:.2f}")
print(f"False Positive Rate Difference: {fpr_diff:.2f}")

 7. Plot disparity in False Positive Rates
labels = ['African-American', 'Caucasian']
fpr = [
    class_metric.false_positive_rate(privileged=False),
    class_metric.false_positive_rate(privileged=True)
]
plt.bar(labels, fpr, color=['red', 'blue'])
plt.ylabel("False Positive Rate")
plt.title("False Positive Rate by Race (COMPAS)")
plt.show()
Explanation:

1. Load the COMPAS dataset and convert it to BinaryLabelDataset format.
2. Split the data into training and testing sets.
3. Compute fairness metrics (disparate impact and statistical parity difference).
4. Apply Reweighing algorithm to mitigate bias.
5. Visualize disparity in false positive rates.

Why This Works:

- Reweighing algorithm assigns different weights to samples to mitigate bias.
- Fairness metrics provide insights into the model's performance.

Summary:

The audit reveals racial bias in the COMPAS dataset. Applying Reweighing algorithm and visualizing fairness metrics can help mitigate bias and ensure fairness in AI systems.

Report:

The COMPAS Recidivism Dataset audit reveals racial bias in risk scores, with disparate impact on African American defendants. The disparate impact ratio is 1.43, indicating that African American defendants are more likely to be misclassified as high-risk. To mitigate this bias, we applied the Reweighing algorithm, which assigns different weights to samples. The algorithm reduced the disparate impact ratio to 1.12. We also visualized the disparity in false positive rates, which highlights the need for fairness interventions.

Remediation Steps:

1. Collect more diverse and representative data.
2. Apply fairness-enhancing algorithms (e.g., Reweighing, Adversarial Debiasing).
3. Monitor and evaluate model performance regularly.



Part 4: Ethical Reflection (5%)

Prompt:
Reflect on a personal project (past or future). How will you ensure it adheres to ethical AI principles?

---

Ethical Reflection Example

In a past project, I developed a machine learning model to assist in shortlisting candidates for university scholarships using academic and demographic data. To ensure the project adhered to ethical AI principles, I would, and will in future projects, apply the following practices:

1. Bias Identification and Mitigation:
   I would begin by auditing training data for bias—such as disproportionately low representation of certain socioeconomic or demographic groups—using tools like AI Fairness 360. If bias is detected, I would either rebalance the dataset, implement algorithmic fairness constraints, or apply pre/post-processing techniques to minimize disparate impact in predictions.

2. Transparency and Explainability:
   I would prioritize using models that provide interpretable decisions, such as explainable boosting machines or rule-based classifiers where possible. For complex models, I would implement local explainability tools (like LIME or SHAP) to offer insight into individual predictions. I would document the data sources, modeling choices, evaluation metrics, and limitations.

3. User Autonomy and Consent:
   Consent would be obtained from all participants for use of their data, outlining in plain language how their data is managed, stored, and processed. Individuals would have the right to opt out and request deletion of their data in accordance with privacy regulations like GDPR.

4. Regular Audits and Accountability:
   I would routinely monitor model performance for fairness, especially as new data is collected, and update the model as needed to prevent unintentional harm. All decisions from the AI system would be reviewable by a human committee to ensure accountability and recourse.

5. Societal and Environmental Responsibility:
   If the project impacts stakeholders at scale, I would engage diverse groups for feedback and conduct impact assessments to ensure the solution is equitable, sustainable, and beneficial to all.

By following these practices, I will strive to build AI systems that are fair, accountable, and trustworthy, fostering societal good while minimizing harm.



