import os
from data_utils import RAW_ROOT, parse_summary_file

patient = "chb01"
summary_path = os.path.join(RAW_ROOT, patient, f"{patient}-summary.txt")

print("Summary path:", summary_path)
print("Exists:", os.path.exists(summary_path))

record_seizures = parse_summary_file(summary_path)

for rec, intervals in record_seizures.items():
    print(rec, intervals)
