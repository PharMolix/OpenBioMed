# SIDER Side Effect Categories

The SIDER database contains 27 side effect categories. The GraphMVP model predicts
the probability of a molecule causing each category of side effects.

## Category Index

| Index | Category | Description |
|-------|----------|-------------|
| 0 | Hepatobiliary disorders | Liver and bile duct disorders |
| 1 | Metabolism and nutrition disorders | Metabolic conditions, nutritional deficiencies |
| 2 | Product issues | Issues related to drug formulation/packaging |
| 3 | Eye disorders | Visual system disorders |
| 4 | Investigations | Abnormal laboratory test results |
| 5 | Musculoskeletal and connective tissue disorders | Bone, muscle, joint disorders |
| 6 | Gastrointestinal disorders | Digestive system disorders |
| 7 | Social circumstances | Social and personal circumstances |
| 8 | Immune system disorders | Immune-related conditions |
| 9 | Reproductive system and breast disorders | Reproductive health issues |
| 10 | Neoplasms benign, malignant and unspecified | Tumors (benign and malignant) |
| 11 | General disorders and administration site conditions | General symptoms, injection site reactions |
| 12 | Endocrine disorders | Hormonal system disorders |
| 13 | Surgical and medical procedures | Procedure-related complications |
| 14 | Vascular disorders | Blood vessel disorders |
| 15 | Blood and lymphatic system disorders | Blood disorders, lymphatic issues |
| 16 | Skin and subcutaneous tissue disorders | Dermatological conditions |
| 17 | Congenital, familial and genetic disorders | Genetic and inherited conditions |
| 18 | Infections and infestations | Infectious diseases |
| 19 | Respiratory, thoracic and mediastinal disorders | Breathing and lung disorders |
| 20 | Psychiatric disorders | Mental health conditions |
| 21 | Renal and urinary disorders | Kidney and urinary tract disorders |
| 22 | Pregnancy, puerperium and perinatal conditions | Pregnancy-related issues |
| 23 | Ear and labyrinth disorders | Hearing and balance disorders |
| 24 | Cardiac disorders | Heart conditions |
| 25 | Nervous system disorders | Neurological conditions |
| 26 | Injury, poisoning and procedural complications | Accidents, overdoses |

## Interpretation

| Probability | Risk Level | Recommendation |
|-------------|------------|----------------|
| > 0.7 | High | Strong warning needed |
| 0.4 - 0.7 | Moderate | Monitor for side effects |
| < 0.4 | Low | Generally safe |

## Usage in Code

```python
# Get SIDER predictions
sider_output = pipeline.run(molecule=molecule, task="SIDER")
probabilities = eval(sider_output[1][0])  # List of 27 floats

# Map to category names
CATEGORIES = [
    "Hepatobiliary disorders", "Metabolism/nutrition disorders", ...,
]

for i, (name, prob) in enumerate(zip(CATEGORIES, probabilities)):
    print(f"{name}: {prob:.4f}")
```
