import wfdb
import os

# Define the database name and the local directory to save it
db_name = 'mitdb'
save_directory = 'mitdb_data'

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

print(f"Downloading database '{db_name}' to '{save_directory}'...")

# Download all files for the database
try:
    wfdb.dl_database(db_name, dl_dir=save_directory)
    print("Download completed successfully.")
    
    # List the contents to verify
    print("\n--- Verifying downloaded files (showing first 10) ---")
    files = os.listdir(save_directory)
    for f in files[:10]:
        print(f)
    if len(files) > 10:
        print(f"...and {len(files) - 10} more files.")

except Exception as e:
    print(f"An error occurred during download: {e}")
    print("Please check your internet connection and permissions.")
