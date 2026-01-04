import subprocess as sub
import os
import venv
import sys
from subprocess import Popen, PIPE, STDOUT
from time import sleep


def venv_install() -> None:
    venv_dir_name = ".venv"
    venv_dir = os.path.abspath(venv_dir_name)
    print
    print(f"\033[34mAttempting to create virtual environment in: {venv_dir}")

    try:
        # Create the virtual environment
        venv.create(venv_dir, with_pip=True, clear=True, symlinks=False) # symlinks=False for better Windows compatibility
        print(f"\033[32mSuccessfully created virtual environment: {venv_dir_name}")

        # Activate the virtual environment
        if sys.platform == "win32":
            # Windows activation
            activate_script = os.path.join(venv_dir, "Scripts", "activate.bat")
            python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
            pip_executable = os.path.join(venv_dir, "Scripts", "pip.exe")
        else:
            # Unix/Linux/MacOS activation
            activate_script = os.path.join(venv_dir, "bin", "activate")
            python_executable = os.path.join(venv_dir, "bin", "python")
            pip_executable = os.path.join(venv_dir, "bin", "pip")

        # Update the current Python process to use the virtual environment's Python
        os.environ["VIRTUAL_ENV"] = venv_dir
        os.environ["PATH"] = os.path.dirname(python_executable) + os.pathsep + os.environ["PATH"]
        
        #Changing to venv
        sys.executable = python_executable
        sys.prefix = venv_dir

        print(f"Virtual environment activated: {venv_dir_name}")
        print(f"Using Python: {python_executable}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    print("Script finished.")
    sleep(3)


def pip_install() -> None:
    
    print(f"\n\nAttempting to install to all required libraries in",end=' ')
    for i in range(3,0,-1):
        print(str(i),end='',flush=True)
        sleep(0.5)
        print(' . ',end='',flush=True)
        sleep(0.5)
    print()
    try:
        cmd = ["pip", "install" ,"-r", "requirements.txt"]

        with Popen(cmd, stdout=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                print("\033[34m",line, end='')

        if p.returncode != 0:
            raise sub.CalledProcessError(p.returncode, p.args)
        else:
            print("\033[32mInstallation succesfull!\033[0m")
    except Exception as e:
        print(e)    


if __name__ == "__main__":
    venv_install()
    pip_install()