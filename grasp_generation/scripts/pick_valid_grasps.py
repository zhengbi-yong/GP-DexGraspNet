import os

def find_files_not_128_bytes(directory, output_file):
    """
    Find files in the given directory that are not 128 bytes in size and save their names to a text file.

    :param directory: Path to the directory to search files in.
    :param output_file: Path to the text file where names of non-128 byte files will be saved.
    """
    with open(output_file, 'w') as file:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                if size != 128:
                    file.write(filename + '\n')

# Replace 'your_directory_path' with the path of the directory you want to search
directory_path = '/home/sisyphus/GP/GP-DexGraspNet/data/leaphand_graspdata_version1_result01/'
# Replace 'output.txt' with the path where you want to save the text file
output_txt_file = 'output.txt'

find_files_not_128_bytes(directory_path, output_txt_file)