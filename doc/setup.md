On Windows: Please install a recent version of Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

You may need to manually modify the installation and ensure the C++ workload is selected.

On all platforms:

```
py -3.11 -m venv venv
cd venv/Scripts
activate
cd ../..
pip install -r requirements.txt
```
