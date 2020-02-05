# Comments on BGWpy in General #

* To cut down on the number of parameters and parameter reuse, I made extensive use of dictionary unpacking.
* When a task is unfinished, perhaps output a bit of the output file so we have an idea why it's unfinished?
* This workflow is a natural candidate for a DAG implementation.
* The HDF5-ness of a task should be exposed:  because the output structure of `Kernel` (and thus input of `Absorption`) is directly affected by this choice, it affects program logic.
* Based on my reading of the `dir()` output for `absorption_task`, none of actually-interesting output from `Absorption` has an associated `fname` variable, only the stdout output (which is interesting to developers only).  When we add an activity for `Absorption`, we *really* should also add `fname` variables to minimize hard-coding file structure.

# Comment on This Notebook in Particular #

* Needs much more meaningful interactivity.  Right now, it's just "run this calculation, it's successful, let's move on".
    * I'm thinking that we should use this as a demonstration on how to converge a calculation.  We may not be able to finish the convergence cycle in a tutorial-appropriate time on a standard workstation, but we can point them in the right direction.
* The ordering of the Ga and As pseudopotentials is highly suspect.  I had to order them the opposite way to what is expected...
* No explanation of parameters are provided.  Really should explain the important ones.
* I've left out `GWFlow` and `BSEFlow`.  My worry is that people will abuse this and not properly converge calculations.
* Gabriel noted that the structure and pseudopotentials that we use in this tutorial is already pre-packaged with BGWpy and can be loaded using the code fragment below.  I'd like to leave this as an option on the table, but for now we will continue using external data files loaded from disk, as this better conforms to the standard user workflow.

```
from BGWpy.data import pseudo_dir, structure_GaAs, pseudos_GaAs
print(pseudo_dir)
print(pseudos_GaAs)
print(Structure.from_file(structure_GaAs))

```

# Introduction to BerkeleyGW #

Here's where we put a bunch of stuff about BerkeleyGW:

* What scientific problem it solves
* What level of theory it works on
* What values it calculates
* How the overall workflow looks
* Its file-based communication structure
* What inputs the overall workflow needs (here, a structure representable by pymatgen and Quantum ESPRESSO pseudopotential files.)

Only as much information should be provided as is needed as to understand this notebook; everything else can be relegated to hyperlinks to external websites.

We should also stress that the parameters chosen for this notebook were used to allow a calculation to run relatively quickly on a standard workstation and are not even close to converged.

# Running This Notebook #

To run this notebook, we assume that you have the following packages installed:

* Jupyter Notebook (otherwise you wouldn't be read this!)
* Quantum ESPRESSO
* BerkeleyGW
* BGWpy

We also assume that you have a basic knowledge of Python and its terminology.

To run BGWpy, you'll also need the `bin` directories of Quantum ESPRESSO and BerkeleyGW installations located in your `PATH` environment variable.

As with all Python-related projects, we highly recommend that you install BGWpy into its own `conda` or `virtualenv` environment.  For more information, please see TODO.

# Debug Stuff (Optional!) #

The following cell is used to generate information that we'll need, should we have to debug this notebook.  You don't need to run it, but it may be useful to look at for educational purposes.


```python
import sys
import os
import BGWpy.config as defaults

print("Python kernel:\n    {}".format(sys.executable))
print("Python version:\n    {}".format(sys.version))
print("Current working directory:\n    {}".format(os.getcwd()))
print("Use HDF5?:\n    {}".format(defaults.use_hdf5))
print("Use complex version of BerkeleyGW?:\n    {}".format(defaults.flavor_complex))
print("DFT Flavor:\n    {}".format(defaults.dft_flavor))
print("Default MPI settings:\n    {}".format(defaults.default_mpi))
print("Default MPI settings:\n    {}".format(defaults.default_runscript))
print("Paths in $PATH:")
for i in os.environ['PATH'].split(":"):
    print("    {}".format(i))
```

# Load Libraries #

First, we load two external packages which BGWpy uses:  `numpy` and `pymatgen`.


```python
import pymatgen
import numpy as np
```

Next, we load the `Structure` class from the BGWpy package.  This module is responsible for creating geometries that BGWpy can read and manipulate using the `pymatgen` package.


```python
from BGWpy import Structure
```

Next, we load the classes which create and run Quantum ESPRESSO calculations.


```python
from BGWpy import QeScfTask, QeBgwFlow
```

Finally, we load the classes with create and run BerkeleyGW calculations.


```python
from BGWpy import EpsilonTask, SigmaTask, KernelTask, AbsorptionTask
```

Make sure that both the BerkeleyGW and Quantum ESPRESSO binary folders are in the PATH folder

# Create the Structure #

For this tutorial, we'll calculate the many-body properties of the GaAs primitive cell.  All files that you will need have been provided for you in the `Data` subdirectory.

SHOW PICTURE HERE.  (Even better if can play using `pymatgen`...)

Geometries are specified in BGWpy using pymatgen's `Structure` class, which may be imported directly from BGWpy or through pymatgen.

There are a number of ways that we can import geometries into BGWpy using the `Structure` class.  For example, we can load them from a pre-existing CIF file:


```python
structure = Structure.from_file('Data/Structures/GaAs.cif')
print(structure)
```

We can also load them from a previous pymatgen Structure which has been exported to a file in the JSON format:


```python
Structure.from_file('Data/Structures/GaAs.json')
print(structure)
```

We can even use pymatgen to directly create the structure in a Python script:


```python
acell_angstrom =  5.6535
rprim = np.array([[.0,.5,.5],[.5,.0,.5],[.5,.5,.0]]) * acell_angstrom
structure = pymatgen.Structure(
    lattice = pymatgen.core.lattice.Lattice(rprim),
    species= ['Ga', 'As'],
    coords = [3*[.0], 3*[.25]],
    )
print(structure)
```

For more information about pymatgen, please consult its official documentation.

# Generating the Ground State Density #

To begin, we will run a ground state DFT calculation to self-consistency to generate the ground state charge density for the calculation.  This ground state charge density will be fed into all wavefunction calculations in the next step.  We use Quantum ESPRESSO in this notebook, however BerkeleyGW and BGWpy supports a number of other DFT packages.

First, we will create a object of the `QeScfTask` task to prepare the needed variables:


```python
scf_task = QeScfTask(
    dirname='Runs/11-Density',

    structure=structure,
    prefix='GaAs',
    pseudo_dir='Data/Pseudos',
    pseudos=['31-Ga.PBE.UPF', '33-As.PBE.UPF'],

    ngkpt=[2, 2, 2],      # k-points grid
    kshift=[.5, .5, .5],  # k-points shift
    ecutwfc=10.0,       # Wavefunctions cutoff energy

    # These are the default parameters for the MPI runner.
    # Please adapt them to your needs.
    nproc=1,
    nproc_per_node=1,
    mpirun='mpirun',
    nproc_flag='-n',
    nproc_per_node_flag='--npernode',
)
```

As you can see, BGWpy has a number of parameters that you will need to set.  However, many of these parameters are consistent from calculation to calculation, so we'll store them in dictionaries that we can reuse for future steps.

First, a dictionary to store all variables that will be used across all Quantum ESPRESSO calculations:


```python
qe_general_settings = {
    'prefix': 'GaAs',
    'pseudo_dir': 'Data/Pseudos',
    'pseudos': ['33-As.PBE.UPF', '31-Ga.PBE.UPF'], # Ordering issue...
}
```

Next, a dictionary to store the variables which are used only for this particular SCF task:


```python
qe_scf_settings = {
    'ngkpt': [2, 2, 2],
    'kshift': [.5, .5, .5],
    'ecutwfc': 10.0,
}
```

And finally, a dictionary to store the settings related to running calculations with MPI.


```python
mpi_settings = {
    'nproc': 4,
    'mpirun': 'mpirun',
    'nproc_flag': '-n',
    'nproc_per_node_flag': " ", # Set to a single space to omit
    'nproc_per_node': " ", # Set to a single space to omit
}
```

Note that all these dictionaries correspond to arguments for the `QeScfTask`, stored as key/value pairs.  This allows us to use dictionary unpacking (see TODO for more information) to considerably tidy up our code:


```python
scf_task = QeScfTask(
    dirname='Runs/11-Density',
    structure=structure,
    **qe_general_settings,
    **qe_scf_settings,
    **mpi_settings,
)
```

Now that we've created the `QeScfTask` task, we can use the `write` method to write the needed input files to disk:


```python
scf_task.write()
```

If you receive an error message stating that an executable could not be found, you likely do not have the needed BerkeleyGW and Quantum ESPRESSO `bin` folders in your `PATH` environment variable.

Let's take a look at the folder that was created by this task using Jupyter's built-in `!ls` magic command:


```python
!ls 'Runs/11-Density'
```

In our new folder, there are two input files:

* `scf.in`, the input file for Quantum ESPRESSO, and
* `run.sh`, the script that will be used to run Quantum ESPRESSO.

`GaAs.save` is a folder used by Quantum ESPRESSO to store intermediate files.

Now that we've created the needed input files, let's run the `run.sh` script using the `run` method.  Note that this step will take a few seconds, as it will run Quantum ESPRESSO in the background.


```python
scf_task.run()
```

Finally, we can check the status of the calculation using the `report` method.  You should see a message telling you that it's been completed.


```python
scf_task.report()
```

Our calculation creates two files that we'll need for the generation of wavefunction files.  We'll store their filenames to reuse later.


```python
charge_density_fname = scf_task.charge_density_fname
data_file_fname = scf_task.data_file_fname
print("Charge density file name: {}".format(charge_density_fname))
print("Data file name:           {}".format(data_file_fname))
```

# Generating the Wavefunctions #

Now that we've generated the ground state density, we'll used this to generate the wavefunctions that we'll feed into BerkeleyGW.  This may be done with the ` QeBgwFlow` class.  As mentioned in the introduction, we'll need up to 6 different types of wavefunction files.

## WFN ##

`WFN` is the "standard" k-shifted wavefunction file which is read by the `Epsilon` calculation, and thus is needed for all BerkeleyGW calculations.

It (and all other wavefunction files) are generated using the `QeBgwFlow` class.  The only difference between these wavefunction types are the parameter values used:


```python
wfn_flow = QeBgwFlow(
    dirname='Runs/12-Wfn',
    structure=structure,

    prefix='GaAs',
    pseudo_dir='Data/Pseudos',
    pseudos=['31-Ga.PBE.UPF', '33-As.PBE.UPF'],

    charge_density_fname='Runs/11-Density/GaAs.save/charge-density.dat',
    data_file_fname='Runs/11-Density/GaAs.save/data-file.xml',

    ngkpt=[2, 2, 2],      # k-points grid
    kshift=[.5, .5, .5],  # k-points shift
    ecutwfc=10.0,       # Wavefunctions cutoff energy
    nbnd=9,             # Number of bands

    # These are the default parameters for the MPI runner.
    # Please adapt them to your needs.
    nproc=1,
    nproc_per_node=1,
    mpirun='mpirun',
    nproc_flag='-n',
    nproc_per_node_flag='--npernode'
)
```

As before, we will break up these arguments into sets of dictionaries: the settings common to all wavefunction calculations


```python
qe_wfn_input_files = {
    'charge_density_fname': charge_density_fname,
    'data_file_fname': data_file_fname
}
```

and the arguments specific to the current wavefunction calculation


```python
qe_wfn_settings = {
    'ecutwfc': 10.0,
    'ngkpt': [2, 2, 2],
    'kshift': [.5, .5, .5],
    'nbnd': 9,
}
```

Note that, because we've already set up the `qe_general_settings` and `mpi_settings` dictionaries, we don't need to re-declare them here.

We can now create the instance of the `QeBgwFlow` class:


```python
wfn_flow = QeBgwFlow(
    dirname='Runs/12-Wfn',
    structure=structure,
    **qe_general_settings,
    **qe_wfn_input_files,
    **qe_wfn_settings,
    **mpi_settings,
)
```

As before, we'll write the input files to disc then run the calculation:


```python
wfn_flow.write()
wfn_flow.run()
wfn_flow.report()
```

The output specifies that we've actually run two calculations here:

* a `WFN` calculation where we calculate wavefunctions using Quantum ESPRESSO, and
* `PW2BGW` where we convert the resulting Quantum-ESPRESSO-specific output files into a format readable by BerkeleyGW.  

Unlike in the density case where we ran a single task, here we're running two tasks (`WFN` and `PW2BGW`) in a workflow (hence the name `QeBgwFlow`).

The output of this step is a wavefunction file:


```python
wfn_fname = wfn_flow.wfn_fname
print(wfn_fname)
```

# WFNq #

Next, we'll create `WFNq`, which is the "standard" k-shifted and q-shifted wavefunction file which is read by the `Epsilon` calculation, and thus is needed for all BerkeleyGW calculations.

The only dictionary we need to create is are the settings specific to the `WFNq` wavefunction:


```python
qe_wfnq_settings = {
    'ecutwfc': qe_wfn_settings['ecutwfc'],
    'ngkpt': qe_wfn_settings['ngkpt'],
    'kshift': qe_wfn_settings['kshift'],
    'qshift': [.001, .0, .0],
}
```

And then we can prepare the calculation:


```python
wfnq_flow = QeBgwFlow(
    dirname='Runs/13-Wfnq',
    structure=structure,
    **qe_general_settings,
    **qe_wfn_input_files,
    **qe_wfnq_settings,
    **mpi_settings,
)
```

Create and run it:


```python
wfnq_flow.write()
wfnq_flow.run()
wfnq_flow.report()
```


```python
wfnq_fname = wfnq_flow.wfn_fname
print(wfnq_fname)
```

## Wfn_co ##

Next, we'll create `WFN_co`, which is the wavefunction on a coarser (and unshifted) grid than `WFN`.  This is used by `Sigma`, `Kernel`, and `Absorption`, and thus will be needed by most BerkeleyGW calculations.  we will also use this calculation to generate the ground state density and exchange-correlation energy density that will be used by `Sigma`.

Once again, we set up the dictionary with our needed variables:


```python
qe_wfn_co_settings = {
    'ecutwfc': qe_wfn_settings['ecutwfc'],
    'ngkpt': [2, 2, 2],
    'kshift': [.0, .0, .0],
    'nbnd': 9,
    'rhog_flag': True,
}
```

Note that there's a new flag `rhog_flag` which tells `QeBgwFlow` to generate additional density-related files.

Now we can prepare the calculation:


```python
wfn_co_flow = QeBgwFlow(
    dirname = 'Runs/14-Wfn_co',
    structure = structure,
    **qe_general_settings,
    **qe_wfn_input_files,
    **qe_wfn_co_settings,
    **mpi_settings,
)
```

And create and run it:


```python
wfn_co_flow.write()
wfn_co_flow.run()
wfn_co_flow.report()
```

As mentioned before, we also output the ground state density and exechange-correlation energy density in this step in a format suitable for Quantum ESPRESSO:


```python
wfn_co_fname = wfn_co_flow.wfn_fname
vxc_dat_fname = wfn_co_flow.vxc_dat_fname
rho_fname = wfn_co_flow.rho_fname
print(wfn_co_fname)
print(vxc_dat_fname)
print(rho_fname)
```

## WFN_fi ##

Next, we'll create `WFN_fi`, the k-shifted `WFN` on a finer grid than `WFN`.  This is used during interpolation in the `Absorption` executable and thus is only needed if you need to solve the BSE equations.  (Symmetry is also turned off for this calculation.)

By this point, you're probably familiar with the steps, so we'll present them without comment:


```python
qe_wfn_fi_settings = {
    'ecutwfc': qe_wfn_settings['ecutwfc'],
    'ngkpt': [2, 2, 2],
    'kshift': [.5, .5, .5],
    'nbnd': 9,
    'symkpt': False,
}
```


```python
wfn_fi_flow = QeBgwFlow(
    dirname = 'Runs/15-Wfn_fi',
    structure = structure,
    **qe_general_settings,
    **qe_wfn_input_files,
    **qe_wfn_fi_settings,
    **mpi_settings,
)
```


```python
wfn_fi_flow.write()
wfn_fi_flow.run()
wfn_fi_flow.report()
```


```python
wfn_fi_fname = wfn_fi_flow.wfn_fname
print(wfn_fi_fname)
```

## WFNq_fi ##

FINALLY, we'll create `WFNq_fi`, the k-shifted and q-shifted `WFN` on a finer grid than `WFN`.  Like `WFN_fi`, this is used during interpolation in the `Absorption` executable and thus is only needed if you need to solve the BSE equations.  (And symmetry is turned off, as before.)

Let's go through the steps again:


```python
qe_wfnq_fi_settings = {
    'ecutwfc': qe_wfn_settings['ecutwfc'],
    'ngkpt': qe_wfn_fi_settings['ngkpt'],
    'kshift': qe_wfn_fi_settings['kshift'],
    'nbnd': qe_wfn_fi_settings['nbnd'],
    'symkpt': qe_wfn_fi_settings['symkpt'],
    'qshift': [.001, .0, .0],
}
```


```python
wfnq_fi_flow = QeBgwFlow(
    dirname = 'Runs/16-Wfnq_fi',
    structure = structure,
    **qe_general_settings,
    **qe_wfn_input_files,
    **qe_wfnq_fi_settings,
    **mpi_settings,
)                                                                            
```


```python
wfnq_fi_flow.write()
wfnq_fi_flow.run()
wfnq_fi_flow.report()
```


```python
wfnq_fi_fname = wfnq_fi_flow.wfn_fname
print(wfnq_fi_fname)
```

# Running GW #

Now the moment you've been waiting for, when we actually run a GW calculation!

## Epsilon ##

Our first step is to run an `Epsilon` calculation, where we'll generate the dielectric matrix (to be precise, the inverse of the dielectric matrix.)

Because BerkeleyGW uses a file-based communication system, we'll need to specify the location of the wavefunction files that we previously calculated:


```python
epsilon_input_files = {
    'wfn_fname': wfn_fname,
    'wfnq_fname': wfnq_fname,
}
```

As well as the settings for an `Epsilon` calculation:


```python
epsilon_settings = {
    'ngkpt': qe_wfn_settings['ngkpt'],    #    'ngkpt': [2, 2, 2],
    'qshift': qe_wfnq_settings['qshift'], #    'qshift': [.001, .0, .0],
    'ecuteps': 10.0,
}
```

And then we can prepare the Epsilon calculation using an `EpsilonTask` object (reusing our `mpi_settings` dictionary from before):


```python
epsilon_task = EpsilonTask(
    dirname='Runs/21-Epsilon',
    structure=structure,
    **epsilon_input_files,
    **epsilon_settings,
    **mpi_settings,
)
```

Let's run the calculation:


```python
epsilon_task.write()
epsilon_task.run()
epsilon_task.report()
```

The result of Epsilon are files containing the (inverse) dielectric function:


```python
epsmat_fname = epsilon_task.epsmat_fname
eps0mat_fname = epsilon_task.eps0mat_fname
print(epsmat_fname)
print(eps0mat_fname)
```

## Sigma ##

Now that we've calculated the (inverse) dielectric matrix and needed wavefunctions, we have everything we need to calculate the GW self-energy.  This is done with the `Sigma` executable, which takes as inputs the results from our `WFN_co` and `Epsilon` calculations:


```python
sigma_input_files = {
    'wfn_co_fname': wfn_co_fname,
    'rho_fname': rho_fname,
    'vxc_dat_fname': vxc_dat_fname,
    'eps0mat_fname': eps0mat_fname,
    'epsmat_fname': epsmat_fname,
}
```

Specify the settings:


```python
sigma_settings = {
    'ngkpt': qe_wfn_co_settings['ngkpt'],  # ngkpt': [2,2,2],
    'ibnd_min': 1,           # Minimum band for GW corrections
    'ibnd_max': 8,           # Maximum band for GW corrections
}
```

Prepare the calculation:


```python
sigma_task = SigmaTask(
    dirname='Runs/22-Sigma',
    structure=structure,
    **sigma_input_files,
    **sigma_settings,
    **mpi_settings,
)
```

And finally run it.


```python
# Execution
sigma_task.write()
sigma_task.run()
sigma_task.report()
```

Our main output file from `Sigma` contains the GW-perturbed eigenvalues:


```python
eqp_fname = sigma_task.eqp1_fname
print(eqp_fname)
```

Note that there are actually *two* files output by Sigma with GW-perturbed eigenvalues: `eqp0.dat` and `eqp1.dat`.  The former contains the on-shell solutions and is not recommended for use for quasi-particle calculations; here we use the second (off-shell) file.  For more information, please see the documentation for `Sigma`.

TODO: Come up with something interesting to say.  Perhaps a convergence exercise?

Let's take a look at the output for `Sigma`:


```python
!cat {eqp_fname}
```

Congratulations!  You have successfully ran a BerkeleyGW calculation from start to finish!

# Running BSE#

For those of you that want to go further, BerkeleyGW can calculate excitionic properties on the GW+BSE level of theory.  This is done with the `KernelTask` and `AbsorptionTask` classes.

## Kernel ##

`Kernel` takes in as inputs the results of `WFN_co` and `Epsilon`:


```python
kernel_input_files = {
    'wfn_co_fname': wfn_co_fname,
    'eps0mat_fname': eps0mat_fname,
    'epsmat_fname': epsmat_fname,
}
```

We can specify its settings:


```python
kernel_settings = {
    'ngkpt': qe_wfn_co_settings['ngkpt'],
    'ecuteps': epsilon_settings['ecuteps'],
    'nbnd_val': 4,
    'nbnd_cond': 4,
    # These extra lines will be added verbatim to the input file.
    'extra_lines': [
        'use_symmetries_coarse_grid',
        'screening_semiconductor',
    ],
}
```

Prepare the calculation:


```python
kernel_task = KernelTask(
    dirname='Runs/23-Kernel',
    structure=structure,
    **kernel_input_files,
    **kernel_settings,
    **mpi_settings,
)
```

And finally run it:


```python
kernel_task.write()
kernel_task.run()
kernel_task.report()
```

The output of this step are file(s) containing the BSE kernel matrix elements.  Note that, if your BerkeleyGW calculation is not using HDF5, there will be two files:  `bsedmat` containing the direct contribution to the kernel and `bsexmat` containing the exchange contribution to the kernel.  If you are using HDF5, these two files will be combined into a single `bsemat` file.


```python
if (kernel_task._use_hdf5):
    # TODO:  Needs to be tested!
    bsemat_fname = kernel_task.bsemat_fname
    print(bsemat_fname)
else:
    bsedmat_fname = kernel_task.bsedmat_fname
    bsexmat_fname = kernel_task.bsexmat_fname
    print(bsedmat_fname)
    print(bsexmat_fname)
```

## Absorption ##

The last step in our adventure together is the solution of the BSE equation via the `Absorption` executable.  It has as inputs the results of `WFN_co`, `WFNq_fi`, and `WFN_fi`, as well as all previous BerkleyGW executables `Epsilon`, `Sigma`, and `Kernel`:


```python
absorption_input_files = {
    'wfn_co_fname': wfn_co_fname,
    'wfn_fi_fname': wfn_fi_fname,
    'wfnq_fi_fname': wfnq_fi_fname,
    'eps0mat_fname': eps0mat_fname,
    'epsmat_fname': epsmat_fname,
    'eqp_fname': eqp_fname,
}
```

As previously mentioned, the file names for the BSE matrix elements will depend on whether you're using HDF5 or not:


```python
if (kernel_task._use_hdf5):
    # TODO:  Needs to be tested!
    absorption_input_files['bsemat_fname'] = bsemat_fname
else:
    absorption_input_files['bsexmat_fname'] = bsexmat_fname
    absorption_input_files['bsedmat_fname'] = bsedmat_fname
```

There are... a lot of settings...


```python
absorption_settings = {
    'ngkpt': [2, 2, 2],        # k-points grid
    'nbnd_val': 4,             # Number of valence bands
    'nbnd_cond': 4,            # Number of conduction bands
    'nbnd_val_co': 4,          # Number of valence bands on the coarse grid
    'nbnd_cond_co': 4,         # Number of conduction bands on the coarse grid
    'nbnd_val_fi': 4,          # Number of valence bands on the fine grid
    'nbnd_cond_fi': 4,         # Number of conduction bands on the fine grid
    # These extra lines will be added verbatim to the input file.
    'extra_lines': [
        'use_symmetries_coarse_grid',
        'no_symmetries_fine_grid',
        'no_symmetries_shifted_grid',
        'screening_semiconductor',
        'use_velocity',
        'gaussian_broadening',
        'eqp_co_corrections',
    ],
    # These extra variables will be added to the input file as '{variable} {value}'.
    'extra_variables': {
        'energy_resolution': 0.15,
    },
}
```

But preparing the calculation is as simple as always:


```python
absorption_task = AbsorptionTask(
    dirname='Runs/24-Absorption',
    structure=structure,
    **absorption_input_files,
    **absorption_settings,
    **mpi_settings,
)
```

And, at last, we can run it.


```python
absorption_task.write()
absorption_task.run()
absorption_task.report()
```

TODO:  Output something cute here.  Perhaps use matplotlib to plot the absorption spectrum?


```python
list(filter(lambda x: "fname" in x, dir(absorption_task)))
```

Congratulations yet again!  You've run a full GW+BSE calculation!m
