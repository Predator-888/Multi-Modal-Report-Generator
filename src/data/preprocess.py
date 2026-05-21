import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"

def setup_directories():
    """Create directory structure if it doesn't exist."""
    for path in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR]:
        os.makedirs(path, exist_ok=True)
        print(f"Verified directory: {path}")

def download_file(url, output_path):
    """Download a file with a progress bar."""
    if os.path.exists(output_path):
        print(f"File already exists at: {output_path}. Skipping download.")
        return

    print(f"Downloading {url} to {output_path}...")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(url)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print("Download completed successfully!")

def extract_tgz(filepath, extract_path):
    """Extract .tgz files to the target directory."""
    print(f"Extracting {filepath} to {extract_path}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found for extraction: {filepath}")
    
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Extraction completed!")

def parse_xml_reports(reports_dir, images_dir):
    """
    Parse the IU X-ray XML reports and map them to their corresponding image filenames.
    Returns a list of dictionaries with clinical features and matched images.
    """
    print("Parsing XML reports...")
    parsed_data = []
    xml_files = [f for f in os.listdir(reports_dir) if f.endswith(".xml")]

    for file_name in tqdm(xml_files, desc="Parsing XMLs"):
        file_path = os.path.join(reports_dir, file_name)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Unique report ID
            report_id = root.find(".//uids").attrib.get("id") if root.find(".//uids") is not None else file_name.replace(".xml", "")

            # Extract report text sections
            findings = ""
            impression = ""
            indication = ""

            for abstract_text in root.findall(".//abstractText"):
                label = abstract_text.attrib.get("label", "").upper()
                text = abstract_text.text if abstract_text.text else ""
                
                if label == "FINDINGS":
                    findings = text.strip()
                elif label == "IMPRESSION":
                    impression = text.strip()
                elif label == "INDICATION":
                    indication = text.strip()

            # Find all associated images
            images = []
            for parent_image in root.findall(".//parentImage"):
                image_id = parent_image.attrib.get("id")
                if image_id:
                    # The image filename in the extracted archive is image_id.png
                    image_filename = f"{image_id}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    # Verify image exists (or just register the expected filename)
                    images.append(image_filename)

            # If there are no images or no findings + impressions, skip to maintain quality
            if not images:
                continue
            if not findings and not impression:
                continue

            # A single report can correspond to multiple images (e.g., frontal and lateral).
            # We will flatten this: each row represents one image with its paired report.
            for img in images:
                parsed_data.append({
                    "report_id": report_id,
                    "image_name": img,
                    "indication": indication,
                    "findings": findings,
                    "impression": impression,
                    "text_report": f"Findings: {findings} Impression: {impression}".strip()
                })

        except Exception as e:
            print(f"Error parsing {file_name}: {e}")

    df = pd.DataFrame(parsed_data)
    print(f"Successfully parsed {len(df)} image-report pairings.")
    return df

def generate_splits(df):
    """Generate train, val, and test splits (80/10/10) and save them to CSV."""
    print("Generating train/val/test splits (80/10/10)...")
    
    # Split based on unique report IDs to prevent data leakage (patient overlapping in different views)
    unique_reports = df["report_id"].unique()
    
    train_ids, test_ids = train_test_split(unique_reports, test_size=0.20, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.50, random_state=42)

    df_train = df[df["report_id"].isin(train_ids)].copy()
    df_val = df[df["report_id"].isin(val_ids)].copy()
    df_test = df[df["report_id"].isin(test_ids)].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    combined_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    # Save splits
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path = os.path.join(SPLITS_DIR, "val.csv")
    test_path = os.path.join(SPLITS_DIR, "test.csv")
    all_path = os.path.join(SPLITS_DIR, "metadata.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    combined_df.to_csv(all_path, index=False)

    print(f"Splits saved successfully!")
    print(f" - Train samples: {len(df_train)}")
    print(f" - Val samples:   {len(df_val)}")
    print(f" - Test samples:  {len(df_test)}")

def run_pipeline(download_images=True):
    """Run the entire preprocessing pipeline."""
    setup_directories()
    
    reports_archive = os.path.join(RAW_DIR, "NLMCXR_reports.tgz")
    images_archive = os.path.join(RAW_DIR, "NLMCXR_png.tgz")
    
    reports_extract_dir = os.path.join(RAW_DIR, "ecir_reports")
    images_extract_dir = os.path.join(RAW_DIR, "images")
    
    # Download XML Reports
    download_file(REPORTS_URL, reports_archive)
    if not os.path.exists(reports_extract_dir):
        os.makedirs(reports_extract_dir, exist_ok=True)
        extract_tgz(reports_archive, reports_extract_dir)

    # Download Images (optional flag if running in a dry-run/local mode)
    if download_images:
        download_file(IMAGES_URL, images_archive)
        if not os.path.exists(images_extract_dir):
            os.makedirs(images_extract_dir, exist_ok=True)
            extract_tgz(images_archive, images_extract_dir)
    else:
        print("Skipping image download. Images must be downloaded separately.")
        os.makedirs(images_extract_dir, exist_ok=True)

    # Parse and split
    df = parse_xml_reports(reports_extract_dir, images_extract_dir)
    if not df.empty:
        generate_splits(df)
        print("Data preprocessing successfully completed!")
    else:
        print("Error: No data was parsed from the XML reports.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IU X-Ray Preprocessing Pipeline")
    parser.add_argument("--skip-images", action="store_true", help="Skip downloading heavy image tgz for testing reports parsing")
    args = parser.parse_args()
    
    run_pipeline(download_images=not args.skip_images)
