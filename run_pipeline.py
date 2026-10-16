import subprocess
import sys

steps = [
    [sys.executable, "src/clean_data.py"],
    [sys.executable, "src/train_models.py"],
]
for step in steps:
    print("\n>>>", " ".join(step))
    subprocess.run(step, check=True)

print("\nPipeline completed successfully.")
print("Run the dashboard with:")
print("streamlit run app/app.py")
