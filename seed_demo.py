"""Build the full 60-record linked synthetic dataset and matching document images."""
from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parent
for script in (ROOT / "scripts" / "build_synthetic_demo.py", ROOT / "scripts" / "regenerate_varied_documents.py", ROOT / "scripts" / "generate_clean_forms.py", ROOT / "scripts" / "export_demo_dataset.py"):
    subprocess.run([sys.executable, str(script)], check=True)
print("Seeded 60 fictional customers, accounts, identity references, bank-card references, varied rotated passbooks, Aadhaar-style and PAN-style synthetic mockups, bank-card mockups, and clean form templates.")
