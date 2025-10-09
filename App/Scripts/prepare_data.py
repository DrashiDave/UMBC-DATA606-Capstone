import os
import zipfile
import pandas as pd

def get_processed_files(log_file):
    """Reads the log file and returns a set of processed filenames."""
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f)

def log_processed_file(log_file, filename):
    """Appends a filename to the log file."""
    if not os.path.exists(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(filename + '\n')

def upsert_csv_from_zip(data_folder='../../Data', output_csv='../../Data/combined_flight_data.csv', log_file='../../Data/Logs/processed_files.log'):    
    """
    Scans a folder for new zip files, extracts the CSV from each,
    and appends the data to a combined CSV file.
    """
    processed_files = get_processed_files(log_file)
    new_files_processed = False
    
    # Ensure the data folder exists
    if not os.path.isdir(data_folder):
        print(f"Error: Data folder '{data_folder}' not found.")
        return

    # Find new zip files to process
    zip_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.zip')])
    
    for item in zip_files:
        if item not in processed_files:
            new_files_processed = True
            print(f"Processing new file: '{item}'")
            zip_path = os.path.join(data_folder, item)
            extract_folder = os.path.join(data_folder, item.replace('.zip', ''))
            
            # Unzip the file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
                print(f"Extracted '{zip_path}' to '{extract_folder}'")

            # Find and process the CSV file
            for root, _, files in os.walk(extract_folder):
                for file in files:
                    if file.endswith('.csv'):
                        csv_path = os.path.join(root, file)
                        try:
                            # Process in chunks to save memory
                            chunk_size = 50000
                            is_first_chunk = not os.path.exists(output_csv)
                            
                            for chunk in pd.read_csv(csv_path, encoding='ISO-8859-1', low_memory=False, chunksize=chunk_size):
                                if is_first_chunk:
                                    chunk.to_csv(output_csv, index=False)
                                    is_first_chunk = False
                                    print(f"Created '{output_csv}' with the first chunk from '{csv_path}'")
                                else:
                                    chunk.to_csv(output_csv, mode='a', header=False, index=False)
                            
                            print(f"Appended data from '{csv_path}' to '{output_csv}'")
                            log_processed_file(log_file, item)
                        except Exception as e:
                            print(f"Could not process {csv_path}: {e}")
    
    if not new_files_processed:
        print("No new files to process.")

if __name__ == "__main__":
    upsert_csv_from_zip()
