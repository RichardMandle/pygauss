from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np
import re
import json
import matplotlib.pyplot as plt

'''
Lots of useful functions for interacting with Gaussian from Python.

Functions:
Minimum_Energy_Conformer:           basic conformer search with rdkit
WriteGauss:                         writes a gaussian input file from smiles input
MakeArcJob:                         create a .sh file for submitting jobs to ARC3/4
read_spectra_from_gaussian_output:  reads IR/Raman data from the .log file
plot_simulated_spectrum:            Plots a simulated FTIR and/or Raman spectrum
'''

def Minimum_Energy_Conformer(smiles, num_of_conformer=500, max_iter=1000, min_energy_MMFF=10000):
    """
    Generate and optimize conformers for a given SMILES string using the MMFF force field and return the conformer
    with the minimum energy.

    Args:
        smiles (str): The input SMILES string.
        num_of_conformer (int): The number of conformers to generate (default: 500).
        max_iter (int): The maximum number of optimization iterations (default: 1000).
        min_energy_MMFF (float): The initial minimum energy value (default: 10000).

    Returns:
        rdkit.Chem.rdchem.Mol: The molecule object representing the conformer with the minimum energy.
    """

    # Create a molecule object from the SMILES string
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens to the molecule
    mol_h_MMFF = Chem.AddHs(mol)

    # Initialize variables for storing minimum energy conformer information
    min_energy_index_MMFF = 0
    min_energy_MMFF = min_energy_MMFF

    # Generate conformers (stored inside the molecule object)
    cids = AllChem.EmbedMultipleConfs(mol_h_MMFF, numConfs=num_of_conformer, params=AllChem.ETKDG())

    # Optimize conformers using the MMFF force field
    results_MMFF = AllChem.MMFFOptimizeMoleculeConfs(mol_h_MMFF, maxIters=max_iter)

    # Search for the conformer with the minimum energy
    print("\nSearching conformers by MMFF")
    for index, result in enumerate(results_MMFF):
        if min_energy_MMFF > result[1]:
            min_energy_MMFF = result[1]
            min_energy_index_MMFF = index
            print(min_energy_index_MMFF, ":", min_energy_MMFF)

    # Create a molecule object representing the conformer with the minimum energy
    mol_min = Chem.Mol(mol_h_MMFF, False, min_energy_index_MMFF)

    # Draw and display the minimum energy conformer
    Draw.MolToImage(mol_min)

    return mol_min
    
    

def WriteGauss(smiles, file_name='smiles', functional='AM1',  basis='', options='', num_of_conformer=500, max_iter=1000, min_energy_MMFF=10000):
    """
    Prepare a Gaussian input file from the given SMILES string using RDKit to generate initial coordinates.

    Args:
        smiles (str): The input SMILES string.
        file_name (str): The name of the output Gaussian input file (default: 'smiles').
        functional (str): The Gaussian functional to use (default: 'AM1').
        basis (str): The basis set to use (default: '').
        options (str): Additional options to include in the Gaussian input file (default: '').

    Returns:
        str: The name of the generated Gaussian input file.
    """
    
    file_name = file_name + '.gjf'  # Add file extension to the output Gaussian input file
    
    mol = Minimum_Energy_Conformer(smiles, num_of_conformer, max_iter, min_energy_MMFF)  # Generate the minimum energy conformer using the previous function
    
    with open(file_name, 'w') as f:
        f.write('%nprocshared=1\n')
        f.write('%mem=2GB\n')
        f.write('#p opt freq polar' + functional + ' ' + basis + ' ' + options + '\n\n')
        f.write(smiles + '\n\n')
        f.write('0 1\n')
        
        for i, atom in enumerate(mol.GetAtoms()):
            positions = mol.GetConformer().GetAtomPosition(i)
            f.write(atom.GetSymbol() + '\t' + str(positions.x) + '\t' + str(positions.y) + '\t' + str(positions.z))
            f.write('\n')
    
    return file_name

    
def MakeARCJob(filename='noname', vmem='8G', nproc=8, startjob=0, endjob=0):
    """
    Generate a job script for running a Gaussain job on ARC (the UoL compute clusters).
    
    Args:
        filename (str): The name of the job script file (default: 'noname').
        vmem (str): The virtual memory allocation for each job (default: '8G').
        nproc (int): The number of processors to use for each job (default: 8).
        startjob (int): The starting index of the task array (default: 0).
        endjob (int): The ending index of the task array (default: 0).

    Returns:
        None
    """
    
    # Decide if we have a task array (i.e. multiple .gjf files) 
    if startjob + endjob == 0:
        Multiple = False
        
    if startjob + endjob != 0:
        Multiple = True
        
        
    with open(filename + '.sh', 'w') as f:
        f.write('#$ -cwd \n')
        f.write('#$ -V\n')
        f.write('#$ -l h_rt=48:00:00\n')
        f.write('#$ -l h_vmem=' + str(vmem) + '\n')
        f.write('#$ -pe smp ' + str(nproc) + '\n')
        f.write('#$ -m be\n')
        f.write('#$ -M r.mandle@leeds.ac.uk\n')
        f.write('#$ -l disk=5G\n')
        if Multiple:
            f.write('#$ -t ' + str(startjob) + '-' + str(endjob) + '\n')  # Create a task array

        f.write('module add gaussian\n')
        f.write('export GAUSS_SCRDIR=$TMPDIR\n')
        f.write('g09 ' + filename + ('_[$SGE_TASK_ID]' * Multiple) + '.gjf\n')
        
        
# functions for reading output
def read_spectra_from_gaussian_output(file_path):
    """
    Read FTIR and Raman spectra data from a Gaussian output file.

    Args:
        file_path (str): The path to the Gaussian output file.

    Returns:
        dict: A dictionary containing the FTIR and Raman spectra data, if available.
    """
    spectra_data = {}

    with open(file_path, 'r') as file:
        contents = file.read()

        # Extract FTIR spectrum data
        ftir_start = contents.find('Harmonic frequencies (cm**-1), IR intensities (KM/Mole)')
        ftir_end = contents.find(' - Thermochemistry -')
        
        if ftir_start != -1 and ftir_end != -1:
            ftir_wavenumbers = []
            ftir_intensities = []
            
            ftir_section = contents[ftir_start:ftir_end].strip().split('\n')[2:-2]

            for line in ftir_section:
                if 'Frequencies' in line:
                    ftir_wavenumbers += [float(freq.strip()) for freq in (line.split("--")[1]).split() if freq.strip()]
                
                if 'IR Inten' in line: 
                    ftir_intensities += [float(freq.strip()) for freq in (line.split("--")[1]).split() if freq.strip()]

            spectra_data['FTIR'] = {'Wavenumbers': ftir_wavenumbers, 'Intensities': ftir_intensities}
            print("FTIR data found")
        else:
            print("No FTIR data found")

        # Extract Raman spectrum data
        
        raman_start = contents.find('Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering')
        raman_end = contents.find('- Thermochemistry -')
        if raman_start != -1 and raman_end != -1 and ('Raman Activ') in contents:
        
            # initialise lists
            raman_wavenumbers = []
            raman_intensities = []
            raman_depolar_p = []
            raman_depolar_u = []
            
            raman_section = contents[raman_start:raman_end].strip().split('\n')[2:]
            
            for line in raman_section:
                if 'Frequencies' in line:
                    raman_wavenumbers += [float(freq.strip()) for freq in (line.split("--")[1]).split() if freq.strip()]
                
                if 'Raman Activ' in line: 
                    raman_intensities += [float(freq.strip()) for freq in (line.split("--")[1]).split() if freq.strip()]

                if 'Depolar (P)' in line: 
                    raman_depolar_p += [float(freq.strip()) for freq in (line.split("--")[1]).split() if freq.strip()]
                
                if 'Depolar (U)' in line: 
                    raman_depolar_u += [float(freq.strip()) for freq in (line.split("--")[1]).split() if freq.strip()]

            spectra_data['Raman'] = {
                'Wavenumbers': raman_wavenumbers,
                'Intensities': raman_intensities,
                'Depolar-P': raman_depolar_p,
                'Depolar-U': raman_depolar_u
            }
            print("Raman data found")
        else:
            print("No Raman data found")

    return spectra_data
    

def plot_simulated_spectrum(spectra_data, fwhm=4):
    """
    Plot a simulated spectrum using the FTIR and Raman data on separate subplots.

    Args:
        spectra_data (dict): Dictionary containing the FTIR and Raman spectra data.
        fwhm (float): Full Width at Half Maximum (FWHM) of the Gaussian curve (default = 4).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot FTIR spectrum
    if 'FTIR' in spectra_data:
        ftir_data = spectra_data['FTIR']
        ftir_wavenumbers = np.array(ftir_data['Wavenumbers'])
        ftir_intensities = np.array(ftir_data['Intensities'])

        # Generate x-axis values (wavenumbers) with finer resolution
        x = np.linspace(ftir_wavenumbers.min(), ftir_wavenumbers.max(), num=5000)

        # Generate y-axis values (intensity) by simulating Gaussian peaks
        y = np.zeros_like(x)
        for wavenumber, intensity in zip(ftir_wavenumbers, ftir_intensities):
            gaussian = intensity * np.exp(-0.5 * ((x - wavenumber) / (fwhm / 2.35482))**2)
            y += gaussian

        # Plot the FTIR spectrum
        ax1.plot(x, y/np.max(y), label='FTIR')
        ax1.set_ylabel('Intensity')

    # Plot Raman spectrum
    if 'Raman' in spectra_data:
        raman_data = spectra_data['Raman']
        raman_wavenumbers = np.array(raman_data['Wavenumbers'])
        raman_intensities = np.array(raman_data['Intensities'])

        # Generate x-axis values (wavenumbers) with finer resolution
        x = np.linspace(raman_wavenumbers.min(), raman_wavenumbers.max(), num=5000)

        # Generate y-axis values (intensity) by simulating Gaussian peaks
        y = np.zeros_like(x)
        for wavenumber, intensity in zip(raman_wavenumbers, raman_intensities):
            gaussian = intensity * np.exp(-0.5 * ((x - wavenumber) / (fwhm / 2.35482))**2)
            y += gaussian

        # Plot the Raman spectrum
        ax2.plot(x, y/np.max(y), label='Raman')
        ax2.set_xlabel('Wavenumber')
        ax2.set_ylabel('Intensity')

    # Add legend
    ax1.legend()
    ax2.legend()

    # Adjust subplot spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    