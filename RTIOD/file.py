import os, os.path

# simple version for working with CWD
print(len([name for name in os.listdir('.') if os.path.isfile(name)]))

# path joining version for other paths
DIR = '/raid/ai24mtech12009/cv_proj/RTIOD/dataset/RTIOD/starting_kit/data/frames'
print(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
