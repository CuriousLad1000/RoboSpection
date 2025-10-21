import os
import re
import shutil

# Current working directory
folder_path = os.getcwd()

# Folder to store backups
backup_folder = os.path.join(folder_path, "sdf_backups")
os.makedirs(backup_folder, exist_ok=True)

old_path_prefix = "/home/user/RoboSpection"
new_path_prefix = "/root/RoboSpection"

# Regex to match <uri>...</uri> containing .stl
uri_pattern = re.compile(r'(<uri>)(.*?\.stl)(</uri>)')

for filename in os.listdir(folder_path):
    if filename.endswith(".sdf"):
        file_path = os.path.join(folder_path, filename)
        
        # Backup file path
        backup_path = os.path.join(backup_folder, filename + ".bak")
        shutil.copyfile(file_path, backup_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Replace old path with new path only in STL URIs
        def replace_uri(match):
            full_path = match.group(2)
            if full_path.startswith(old_path_prefix):
                full_path = full_path.replace(old_path_prefix, new_path_prefix, 1)
            return match.group(1) + full_path + match.group(3)
        
        new_content = uri_pattern.sub(replace_uri, content)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print(f"Updated STL URIs in {filename} (backup saved in {backup_folder})")
