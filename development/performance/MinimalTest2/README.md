# Simple timing test of integrators with and without switching off a water molecule in a water box

To run:
```bash
# Make sure you are using an OpenMM 7.1.0 preview build
conda install --yes -c omnia/labels/dev openmm
# Install openmmtools to get access to testsystems
conda install --yes -c omnia openmmtools
# Run the timing comparison
python minimal_test.py
```

