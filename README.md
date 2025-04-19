# quantum-hypertuning
CSCI 4965 / 6995 Final Project 

There is a custom patch for the qiskit_algorithms package, you should be able to run the 5 commands below without issue, but if for some reason it doesnt work,
I also included manual instructions for the patch

Here is the sequence of commnads to get the reward function working.

```
user@machine:~/quantum-hypertuning$ python3 -m venv project-venv
user@machine:~/quantum-hypertuning$ pip install --upgrade pip setuptools wheel
user@machine:~/quantum-hypertuning$ pip install -r requirements.txt
user@machine:~/quantum-hypertuning$ rm project-venv/lib/python3.12/site-packages/qiskit_algorithms/amplitude_estimators/iae.py
user@machine:~/quantum-hypertuning$ cp ./custom_iae.py project-venv/lib/python3.12/site-packages/qiskit_algorithms/amplitude_estimators/iae.py
```
If for some reason the patch doesnt work and you start getting either KeyErrors or AttributeErrors from the libs, you might have to do it manually:

in project-venv/lib/python3.12/site-packages/qiskit_algorithms/amplitude_estimators/iae.py, on line 317, replace "shots = ret.metadata[0].get("shots")" with:

metadata = ret.metadata[0] if isinstance(ret.metadata, list) and ret.metadata else {}
shots = metadata.get("shots", 1000)


