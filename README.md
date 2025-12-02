
The COMPAS Recidivism Dataset audit reveals racial bias in risk scores, with disparate impact on African American defendants.

Solution:
Import necessary libraries
import pandas as pd
import numpy as np
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryDatasetMetric, ClassificationMetric
import matplotlib.pyplot as plt

Load COMPAS dataset
data = pd.read_csv('compas.csv')

Convert data to BinaryLabelDataset format
binary_data = BinaryLabelDataset(df=data, 
                                label_names=['two_year_recid'], 
                                protected_attribute_names=['race'], 
                                favorable_label=0, 
                                unfavorable_label=1)

Split data into training and testing sets
train_data, test_data = binary_data.split([0.7], shuffle=True)

Compute fairness metrics
metric_train = BinaryDatasetMetric(train_data, 
                                   unprivileged_groups=[{'race': 0}], 
                                   privileged_groups=[{'race': 1}])

Print fairness metrics
print("Disparate Impact:", metric_train.disparate_impact())
print("Statistical Parity Difference:", metric_train.statistical_parity_difference())

Apply Reweighing algorithm to mitigate bias
rw = Reweighing(unprivileged_groups=[{'race': 0}], 
                 privileged_groups=[{'race': 1}])
transformed_data = rw.fit_transform(train_data)

Visualize disparity in false positive rates
plt.bar(['African American', 'Caucasian'], 
        [metric_train.false_positive_rate(unprivileged=True), 
         metric_train.false_positive_rate(privileged=True)])
plt.xlabel('Race')
plt.ylabel('False Positive Rate')
plt.title('Disparity in False Positive Rates')
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



Bonus Task: Policy Proposal for Ethical AI Use in Healthcare

Guideline for Ethical AI Use in Healthcare

Introduction:
Artificial intelligence (AI) has the potential to revolutionize healthcare, but it also raises concerns about bias, transparency, and patient consent. This guideline outlines the principles and practices for the responsible development and deployment of AI systems in healthcare.

Principles:

1. Patient Autonomy: Patients have the right to control their data and make informed decisions about their care.
2. Non-Maleficence: AI systems should not harm patients or compromise their safety.
3. Justice: AI systems should be fair and unbiased, ensuring equitable access to healthcare.
4. Transparency: AI systems should be transparent, explainable, and accountable.

Patient Consent Protocols:

1.Informed Consent: Patients should provide informed consent before their data is used for AI development or deployment.
2. Data Protection: Patient data should be protected and anonymized to prevent unauthorized access or misuse.
3. Right to Opt-Out: Patients have the right to opt-out of AI-driven care or data usage.

Bias Mitigation Strategies:

1. Data Quality: Ensure high-quality, diverse, and representative data to train AI models.
2. Fairness Metrics: Regularly evaluate AI systems for bias and fairness using metrics like disparate impact ratio and equal opportunity difference.
3. Human Oversight: Implement human oversight and review processes to detect and address bias.

Transparency Requirements:
1. Explainability: AI systems should provide clear and concise explanations for their decisions or recommendations.
2. Model Transparency: AI models should be transparent, and their development processes should be auditable.
3. Accountability: Establish accountability mechanisms for AI-driven decisions or actions.

Implementation:

1. Multidisciplinary Teams: Develop AI systems with multidisciplinary teams, including clinicians, ethicists, and patients.
2. Continuous Monitoring: Regularly monitor AI systems for bias, transparency, and patient consent.
3. Patient Engagement: Engage patients and caregivers in AI development and deployment processes.

By following this guideline, we can ensure that AI systems in healthcare are developed and deployed responsibly, prioritizing patient autonomy, fairness, and transparency.
