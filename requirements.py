import subprocess as sub
import os
import venv
import sys


def venv_install() -> None:
    venv_dir_name = ".venv"
    venv_dir = os.path.abspath(venv_dir_name)

    print(f"Attempting to create virtual environment in: {venv_dir}")

    try:
        # Create the virtual environment
        venv.create(venv_dir, with_pip=True, clear=True, symlinks=False) # symlinks=False for better Windows compatibility
        print(f"Successfully created virtual environment: {venv_dir_name}")

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


def pip_install() -> None:
    try:
        result = sub.run(
        "pip install -r requirements.txt",
        capture_output=True,
        text=True,
        shell=True
        )
        if not result.stdout:
            print("Not found")
        else:
            print(result.stdout)
    except Exception as e:
        print(e)    


if __name__ == "__main__":
    venv_install()
    pip_install()