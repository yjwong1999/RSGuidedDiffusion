# My Kaggle authentication
# {"username":"yjwong99","key":"5708a20c5643d05827b398dce05031bc"}

#---------------------------------------------------------
# Import dependencies
#---------------------------------------------------------
import opendatasets as od
import os, shutil, subprocess


# run command
def run_command(command):
    """
    Execute a command in the shell.
    """
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


# get current working directory (to make sure the code works anywhere in your device)
cwd = os.getcwd()


#---------------------------------------------------------
# Download the raw data from Kaggle
#---------------------------------------------------------
# download the data
print('\n#-----------------------------------------------------------------------------------')
print('Download the Building Extraction Generalization 2024 Dataset from kaggle')

od.download("https://www.kaggle.com/competitions/building-extraction-generalization-2024/data")


#---------------------------------------------------------
# Restructure the dataset
#---------------------------------------------------------
print('\n#-----------------------------------------------------------------------------------')
print('Restructuring the dataset into directory "detect" for segmentation/diffusion')
# remove directory if exist
if os.path.exists('detect'):
    shutil.rmtree('detect') 

# create directory
os.makedirs('detect/train/image')
os.makedirs('detect/train/label')

os.makedirs('detect/val/image')
os.makedirs('detect/val/label')

# convert from coco-seg format to yolo-seg format
command = f'python3 coco2yolo/coco2yolo -ann-path "{cwd}/building-extraction-generalization-2024/train/train.json" -img-dir "{cwd}/building-extraction-generalization-2024/train" -task-dir "{cwd}/detect/train" -set union'
run_command(command)

command = f'python3 coco2yolo/coco2yolo -ann-path "{cwd}/building-extraction-generalization-2024/val/val.json" -img-dir "{cwd}/building-extraction-generalization-2024/val" -task-dir "{cwd}/detect/val" -set union'
run_command(command)


# Source and destination directories
cwd = os.getcwd()
src_dir = f"{cwd}/detect/train/image"
dst_dir = f"{cwd}/detect/train/label"

# Iterate through files in the source directory
for filename in os.listdir(src_dir):
    if filename.endswith('.txt'):
        # Construct full file paths
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # Copy the file
        shutil.copy(src_path, dst_path)

        # Delete the file from the source directory
        os.remove(src_path)


# Source and destination directories
cwd = os.getcwd()
src_dir = f"{cwd}/detect/val/image"
dst_dir = f"{cwd}/detect/val/label"

# Iterate through files in the source directory
for filename in os.listdir(src_dir):
    if filename.endswith('.txt'):
        # Construct full file paths
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # Copy the file
        shutil.copy(src_path, dst_path)

        # Delete the file from the source directory
        os.remove(src_path)

#---------------------------------------------------------
#
#---------------------------------------------------------
print('\n#-----------------------------------------------------------------------------------')
print('Restructuring the dataset into directory "mydata" for YOLOv8 training')
