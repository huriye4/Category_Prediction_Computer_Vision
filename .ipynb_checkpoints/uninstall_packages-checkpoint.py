import subprocess

with open('requirements.txt', 'r') as file:
    packages = file.readlines()
    packages = [pkg.strip() for pkg in packages]

for package in packages:
    subprocess.call(['pip', 'uninstall', '-y', package])
