import pandas as pd
import numpy as np
import random

def generate_sample_batch(filename, num_samples=150):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    for i in range(num_samples):
        # Base attributes
        age = np.random.randint(18, 80)
        room = random.choice(["SAPHIRE", "ZAMRUD", "BERLIAN"])
        
        # Determine case type
        case_type = random.choice(["Normal", "Normal", "Normal", "Fever_Inflated", "Incomplete_Doc"])
        
        if case_type == "Normal":
            icd = random.choice(["E11.5", "A91", "I10"])
            los = np.random.randint(3, 7)
            cost = np.random.randint(2_000_000, 5_000_000)
            resume = 1
            auth = 1
        elif case_type == "Fever_Inflated":
            icd = "R50.9" # Fever
            los = np.random.randint(1, 3)
            cost = np.random.randint(6_000_000, 10_000_000) # Abnormal cost for fever
            resume = 1
            auth = 1
        else: # Incomplete_Doc
            icd = random.choice(["E11.5", "A91", "I10"])
            los = np.random.randint(2, 5)
            cost = np.random.randint(2_000_000, 5_000_000)
            resume = 0
            auth = 0
            
        data.append({
            "claim_id": f"CLM-Q2-{2000 + i}",
            "patient_age": age,
            "room_type": room,
            "icd_10_code": icd,
            "total_cost": cost,
            "is_resume_complete": resume,
            "auth_signature": auth,
            "los": los
        })
        
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"File {filename} generated with {len(df)} rows.")

if __name__ == "__main__":
    generate_sample_batch("sample_batch_claims_Q2.csv")
