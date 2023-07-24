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
Minimum_Energy_Conformer			-	Generates the minimum energy conformer of a given molecule (as smiles) using the MMFF94 method
WriteGauss							-	Takes a smiles string, performs energy minimisation, then writes the 3D coordinates to a gaussian input file
MakeARCJob							- 	Creates a .sh file for launching a job on the SGE nodes of ARC3/ARC4
read_spectra_from_gaussian_output	-	Reads spectral data (IR, Raman) from a Gaussian output file
read_complexation_energy			-	Reads the complexation energies from a gaussian calculation with counterpoise correction
extract_properties					-	Extracts properties from a gaussian output file (e.g. dipole, polarisability etc.)
plot_simulated_spectrum				-	Plots a simulated FTIR/Raman spectrum from Gaussian data
read_energy							-	Reads the final energy of a molecule in a gaussian calculation
read_gaussian_route_section			-	Reads the route section of a gaussian file to determine what sort of job was performed
read_nmr_data						-	Reads NMR data from a Gaussian output file
get_nmr_correction_factors			-	Returns the NMR correction factors for a GAIO NMR job
do_scaling							-	Scales calculated NMR data according to correction factors
plot_nmr_spectrum					-	Plots a simulated NMR spectrum 
read_tddft_data(file_path)          -   Reads td-dft data from a gaussian output file
plot_uvvis_spectra                  -   Plots a uv-vis spectrum from given td-dft data
read_complexation_energy            -   Returns complexation energies from a gaussian counterpoise calculation
'''

def Minimum_Energy_Conformer(smiles, num_of_conformer=500, max_iter=1000, min_energy_MMFF=10000):
    """
    Generate and optimize conformers for a given SMILES string using the MMFF force field and return the conformer
    with the minimum energy.
    
    This is useful for getting an initial starting point for a gaussian calculation, as it 
    generates a decent 3D geometry from a 2D (or 0D) molecular input.

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
    
    

def WriteGauss(smiles,vmem=8,nproc=8, file_name='smiles', functional='AM1',  basis='', options='', num_of_conformer=500, max_iter=1000, min_energy_MMFF=10000):
    """
    Prepare a Gaussian input file from the given SMILES string using RDKit to generate initial coordinates.

    Args:
        smiles (str): The input SMILES string.
        vmem: The ammount of RAM to use (in GB)
        nproc: the number of cores to use
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
        f.write('%nprocshared=' + str(nproc) + '\n')
        f.write('%mem=' + str(vmem) + 'GB\n')
        f.write('#p opt freq polar ' + functional + ' ' + basis + ' ' + options + '\n\n')
        f.write(smiles + '\n\n')
        f.write('0 1\n')
        
        for i, atom in enumerate(mol.GetAtoms()):
            positions = mol.GetConformer().GetAtomPosition(i)
            f.write(atom.GetSymbol() + '\t' + str(positions.x) + '\t' + str(positions.y) + '\t' + str(positions.z))
            f.write('\n')
        f.write('\n\n')
    return file_name

    
def MakeARCJob(filename='noname', vmem=8, nproc=8, startjob=0, endjob=0):
    """
    Generate a job script for running a Gaussain job on ARC (the UoL compute clusters).
    
    Args:
        filename (str): The name of the job script file (default: 'noname').
        vmem (str): The virtual memory allocation for each job in GB(default: '8').
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
        f.write('#$ -l h_vmem=' + str(vmem) + 'G\n')
        f.write('#$ -pe smp ' + str(nproc) + '\n')
        f.write('#$ -m be\n')
        f.write('#$ -M r.mandle@leeds.ac.uk\n')
        f.write('#$ -l disk=5G\n')
        if Multiple:
            f.write('#$ -t ' + str(startjob) + '-' + str(endjob) + '\n')  # Create a task array

        f.write('module add gaussian\n')
        f.write('export GAUSS_SCRDIR=$TMPDIR\n')
        f.write('g09 ' + filename + ('_$SGE_TASK_ID' * Multiple) + '.gjf\n')
    
    return    
        
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
    
    
def read_complexation_energy(gaussian_output_file='NoName',loud=False):
    """
    This function reads Gaussian output and takes the complexation energies related to 
    a counterpoise calculation in Gaussian
    
    Args:
        gaussian_output_file    - the filename of the gaussianjob to read from
        loud                    - if true, will print the scraped data to the terminal.
    returns:
    a array of data:
     Counterpoise corrected energy
                       BSSE energy
                   sum of monomers
               complexation energy
               complexation energy

    """
    with open(gaussian_output_file, 'r') as file:
        text = file.read()
        
    pattern = r'Counterpoise corrected.*\(corrected\)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        matched_text = match.group(0)
        float_values = re.findall(r'[-+]?\d*\.\d+', matched_text)
        if loud:
            print('Counterpoise corrected energy: ' + str(float_values[0]) + ' Ha')
            print('BSSE energy: ' + str(float_values[1]) + ' Ha')
            print('sum of monomers: ' + str(float_values[2]) + ' Ha')
            print('complexation energy (raw): ' + str(float_values[3]) + ' kcal/mole')
            print('complexation energy (corrected): ' + str(float_values[4]) + ' kcal/mole')
        
        return [float(value) for value in float_values]
              
    else:
        return []    
    
def extract_properties(file_path):
    """
    Tool for extracting various properties from a gaussian output file.
    
    Args:
        file_path (str): The path to the Gaussian output file.

    Returns:
        dict: A dictionary containing the properties, as available.
    """
    
    properties = {} # dict for storing properties.
    
    with open(file_path, 'r') as file:
        contents = file.read()
        
    # check job finished
    if contents.find("Normal termination of Gaussian") == -1:
        #if it didn't, print an error and exit
        print('Job did not successfully complete')
        return
    
    # check for imaginary frequencies:
    if contents.find("imaginary frequencies") != -1:
        print('\n** WARNING**\nimaginary frequencies found!\n')
    
    # extract dipole tensor
    if contents.find("Dipole moment (field-independent basis, Debye):"):
        pattern = r"X=\s+([\d.-]+)\s+Y=\s+([\d.-]+)\s+Z=\s+([\d.-]+)"
        matches = re.findall(pattern, contents)
        
        if matches:
            dipole_values = np.array(matches[0], dtype=float)
            properties["Dipole tensor"] = dipole_values
            #print("Dipole tensor values:", properties.get("Dipole tensor", []))

    # extract quadrupole tensor
    if contents.find("Quadrupole moment (field-independent basis, Debye-Ang):"):
        pattern = r"XX=\s+([\d.-]+)\s+YY=\s+([\d.-]+)\s+ZZ=\s+([\d.-]+)\s+XY=\s+([\d.-]+)\s+XZ=\s+([\d.-]+)\s+YZ=\s+([\d.-]+)"
        matches = re.findall(pattern, contents)

        if matches:
            quadrupole_values = np.array(matches[0], dtype=float)
            properties["Quadrupole tensor"] = quadrupole_values
            #print("Quadrupole tensor values:", properties.get("Quadrupole tensor", []))
            
    
    # extract polarisability tensor
    if contents.find("Exact polarizability"):
        pattern = r"Exact polarizability:\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)"
        matches = re.findall(pattern, contents)

        if matches:
            polarizability_values = [float(match) for match in matches[0]]
            properties["Exact polarizability"] = polarizability_values
            #print("Exact polarizability values:", properties.get("Exact polarizability", []))
    
    if contents.find("Molar volume ="):
        pattern = r"\(\s*([\d.]+)\s*cm\*\*3/mol\)"
        matches = re.findall(pattern, contents)

        if matches:
            molar_volume = float(matches[0])
            properties["Molar Volume"] = molar_volume

    
    return properties
    

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
        ax1.set_xlabel('Wavenumber / cm^-1')
        
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
        ax2.set_xlabel('Wavenumber / cm^-1')
        ax2.set_ylabel('Intensity')

    # Add legend
    ax1.legend()
    ax2.legend()

    # Adjust subplot spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def read_energy(GaussianOutputFile='NoName'):
    """
    This function reads Gaussian output and takes the single-point energy ONLY
    If you've done geometry optimisation this will be ignored!
    
    The code that looks for energy searches for a string ("SCF DONE: E(") - it is 
    not clear how robust this is, but it works for E(RAM1) calculations fine.
    
    Sometimes gaussian gives you .out files, sometimes .log. This is configured
    for .out files, because thats what my laptop is giving me today (17/1/23)
    """

    energy = ['']
    #SCF Done:  E(RAM1) = -0.120657830862     A.U. after    2 cycles
    g09output=open(GaussianOutputFile,'r')
    for line in g09output: # find the line that contains the SCF energy and store
        if 'SCF Done:' in line: # *** possible weakpoint in code ***
            Scraped = str([line.strip('\n')])
            Scraped = Scraped.split('A.U.',1)[0]
            Scraped = Scraped.split(' = ',1)[1]
            energy.append((float(Scraped) * 627.5)) # extract the energy, convert to Kcal mol-1 (1Ha = 627.5 Kcal mol-1)
    g09output.close()
    
    return(energy)
    
    
def read_gaussian_route_section(file_path):
    '''
    Reads the route-section of a Gaussian .log/.out file
    and returns the route options as a list
    
    Args:
        file_path - file path of the .log/.out file to read
        
    Returns:
        input_options - a list of input options
    '''
    
    input_options=[]
    with open(file_path, 'r') as log_file:
        start_flag = False
        for line in log_file:
            line = line.strip()
            if line.startswith('----------------------------------------------------------------------'):
                if start_flag:
                    break  # Stop processing after the first set of input options
                start_flag = True
            elif start_flag:
                options = line.split()
                input_options.extend(options)

    return(input_options) 

def read_nmr_data(file_path):
    '''
    Reads NMR data from a Gaussian .log/.out file
    
    Args:
        file_path - file path of the .log/.out file to read     
    Returns:
        input_options - a list of input options
    nmr_data - a dict of NMR data; chemical shifts, by element
    '''
    nmr_data = {}
    with open(file_path, 'r') as file :
        for line in file :
            line = line.strip()
               
            if 'Isotropic' in line:
                # Parse the line to extract atom number, element, and chemical shift
                parts = line.split()
                atom_number = int(parts[0])
                element = parts[1]
                chemical_shift = float(parts[4])
                
                # Store the data in the dictionary
                nmr_data[atom_number] = {
                    'element': element,
                    'chemical_shift': chemical_shift
                }
      
    return(nmr_data)
    
def get_nmr_correction_factors():
    '''
    NMR correction factors taken from: 
    http://cheshirenmr.info/ScalingFactors.htm#table1aheading
    
    and returned as a dict.
    1st and 3rd values are slope of 1H and 13C, respectively
    2nd and 4th values are intercept of 1H and 13C
    
    dict key is method/basis set.
    '''    
    
    corrections = {
        'B3LYP/6-31G(d)':[-0.9957,32.2884,-0.9269,187.4743],
        'B3LYP/6-31+G(d,p)':[-1.0591,31.6430,-0.9671,190.4279],
        'B3LYP/6-311+G(2d,p)':[-1.0784,31.8723,-1.0537,181.7815],
        'B3LYP/6-311+G(2d,p)':[-1.0781,31.9786,-1.0451,182.3835],
        'B3LYP/aug-cc-pVDZ':[-1.0784,31.6430,-1.0048,190.9490],
        'MP2/6-31+G(d,p)':[-1.0798,31.9282,-0.9358,204.0880],
        'MP2/6-311+G(2d,p)':[-1.0954,32.0110,-1.0130,195.4108],
        'VSXC/aug-cc-pVDZ':[-1.0536,31.2003,-0.9589,184.8018],
        'WC04/aug-cc-pVDZ':[-0.9760,197.7614,0,0],
        'WP04/aug-cc-pVDZ':[-1.0527,31.8773,0,0],
        'mPW1PW91/6-311+G(2d,p)':[-1.0936,31.8018,-1.0533,186.5242],
        'mPW1PW91/6-311+G(2d,p)':[-1.0933,31.9088,-1.0449,187.1018],
        'PBE0/6-311+G(2d,p)':[-1.0958,31.7532,-1.0533,187.3123],
        'PBE0/6-311+G(2d,p)':[-1.0956,31.8603,-1.0450,187.8859],
        'M062X/6-31G(d)':[-1.1298,32.5438,-1.0801,196.9784],
        'M062X/6-31+G(d,p)':[-1.1625,32.0027,-1.0982,198.2814],
        'M062X/6-311+G(2d,p)':[-1.1756,32.2000,-1.1746,189.5548],
        'M062X/6-311+G(2d,p)':[-1.1759,32.3045,-1.1650,189.9794],
        'M06L/6-31G(d)':[-1.0127,32.4569,-0.8684,186.5207],
        'M06L/6-31+G(d,p)':[-1.0564,31.9255,-0.8856,188.7538],
        'M06L/6-311+G(2d,p)':[-1.0645,32.1187,-0.9491,181.4397],
        'M06L/6-311+G(2d,p)':[-1.0662,32.2242,-0.9439,181.9711],
        'mPW1PW91/6-31G(d)':[-1.0401,32.2587,-0.9537,193.2179],
        'mPW1PW91/6-31+G(d,p)':[-1.0803,31.7031,-0.9726,194.9643],
        'mPW1PW91/6-311+G(2d,p)':[-1.0938,31.8723,-1.0446,186.7246],
        'mPW1PW91/6-311+G(2d,p)':[-1.0951, 31.9773,1.0379,187.2065],
        'b3lyp/6-311g(d,p)':[-0.9957,32.2884,-0.9269,187.4743],
              }
    
    return(corrections)
    
def do_scaling(chemical_shift,route,element='H'):
    '''
    scales the NMR data using some appropriate method  
    '''
    
    corrections = get_nmr_correction_factors() # return a dict of corrections
    if element.lower() == 'h': # decide which slope/intercept we need
        Elem = 0
    if element.lower() == 'c':
        Elem = 2
    
    # find the match between route method and this dict
    
    method = ''
    for n in range(len(route)):
        if route[n].lower() in map(str.lower,corrections):
            method = route[n]
            break
            
    if method == '':    
        print('No correction factor found for: ' + method.lower())
        method = 'B3LYP/6-31G(d)'
        
    matched_key = ''
    for key in corrections:
        if(key.lower() == method.lower()):
            matched_key = key
            print('Applying Correction for ' +  matched_key)
            break
                    
    # apply the correction  
    if matched_key in corrections:
        slope = corrections[matched_key][0+Elem]
        intercept = corrections[matched_key][1+Elem]
        chemical_shift = chemical_shift / slope + intercept

    return(chemical_shift)
    
def plot_nmr_spectrum(nmr_data,file_path,element='H',instrument_frequency=400):
    '''
    plots an NMR spectrum
    Args:
        - nmr_data - dict of NMR data generated with read_nmr_data
        - element  - element to plot spectrum of
        - instrument_frequency - the psuedo frequency of the simulated spectra, in MHz
    '''
    chemical_shifts = []
    for atom_number, data in nmr_data.items():
        if data['element'] == element:
            chemical_shifts.append(data['chemical_shift'])
    chemical_shifts = np.array(chemical_shifts)
    print(f"Uncorrected chemical shifts for element {element}:")
    print(chemical_shifts)

    route = read_gaussian_route_section(file_path)
    print(route)
    chemical_shifts = do_scaling(chemical_shifts,route,element)


    fwhm = 37/instrument_frequency

    # Generate x-axis values (wavenumbers) with finer resolution
    x = np.linspace(chemical_shifts.min()-5,
                    chemical_shifts.max()+5, 
                    num=5000)

    # Generate y-axis values (intensity) by simulating Gaussian peaks
    y = np.zeros_like(x)
    for wavenumber, intensity in zip(chemical_shifts, chemical_shifts/chemical_shifts):
        gaussian = intensity * np.exp(-0.5 * ((x - wavenumber) / (fwhm / 2.35482))**2)
        y += gaussian
    
    # Plot the NMR spectrum
    plt.plot(x, y/np.max(y), label='NMR Spectrum')
    plt.xlabel('Chemical Shift / ppm')
    plt.ylabel('I / arb.')
    plt.show()
    
    return

def read_tddft_data(file_path):
    """
    Reads td-dft data from a G09 file.

    Args:
        file_path (str): The path to the Gaussian output file.

    Returns:
        dict: A dictionary containing the TD_DFT data, if available.
    
    Usage:
        uvvis_data = read_tddft_data('TD_DFT_TEST1.LOG')
    """
    spectra_data = {}

    with open(file_path, 'r') as file:
        contents = file.read()

        # Extract FTIR spectrum data
        tddft_start = contents.find('Excitation energies and oscillator strengths:')
        tddft_end = contents.find('SavETr:')
        
        if tddft_start != -1 and tddft_end != -1:
            tddft_wavelength = []
            tddft_strengths = []
            
            tddft_section = contents[tddft_start:tddft_end].strip().split('\n')[2:-2]

            for line in tddft_section:
                if 'Excited State' in line:
                    floats = (re.findall(r"\d+\.\d+", line)) # oscilator data as floats: ev / nm / f / s**2
                    
                    tddft_wavelength.append(floats[1])
                    tddft_strengths.append(floats[2])

            spectra_data['uv-vis'] = {'wavelengths': tddft_wavelength, 'strengths': tddft_strengths}
            print("TD-DFT data found")
            print(spectra_data)
        else:
            print("No TD-DFT data found")

    return spectra_data

def plot_uvvis_spectra(spectra_data, fwhm=50, lambda_min=200, lambda_max=1000):
    """
    makes a uv-vis spectrum from td-dft data
    
        Args:
    spectral_data - dict of spectral data returned by read_tddft_data
    fwhm          - the fwhm of the peaks in the resulting spectrum
    lambda_min    - minimum wavelength range in our spectrum
    lambda_max    - maximum wavelength range of our spectrum
    
        Returns:
    a cool spectrum
    
        Usage:
    plot_uvvis_spectra(uvvis_data,75,200,600)
    
    """
    if 'uv-vis' in spectra_data:
        uvvis_wavelength = np.array(spectra_data['uv-vis']['wavelengths'])
        uvvis_intensities = np.array([float(intensity) for intensity in spectra_data['uv-vis']['strengths']])

        x = np.linspace(lambda_min, lambda_max, num=5000)
        y = np.zeros_like(x)

        for wavelength, intensity in zip(uvvis_wavelength, uvvis_intensities):
            gaussian = intensity * np.exp(-0.5 * ((x - float(wavelength)) / (fwhm / 2.35482))**2)
            y += gaussian

        fig, ax = plt.subplots()
        ax.plot(x, y/np.max(y), label='uv-vis')
        ax.set_ylabel('Oscillator Strength')
        ax.set_xlabel('Wavelength / nm')

        plt.show()
   
    return

def read_complexation_energy(GaussianOutputFile='NoName',loud=False):
    """
    This function reads Gaussian output and takes the complexation energies related to a counterpoise calculation
    
    returns:
     Counterpoise corrected energy
                       BSSE energy
                   sum of monomers
               complexation energy
               complexation energy

    """
    with open(GaussianOutputFile, 'r') as file:
        text = file.read()
        
    pattern = r'Counterpoise corrected.*\(corrected\)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        matched_text = match.group(0)
        float_values = re.findall(r'[-+]?\d*\.\d+', matched_text)
        if loud:
            print('Counterpoise corrected energy: ' + str(float_values[0]) + ' Ha')
            print('BSSE energy: ' + str(float_values[1]) + ' Ha')
            print('sum of monomers: ' + str(float_values[2]) + ' Ha')
            print('complexation energy (raw): ' + str(float_values[3]) + ' kcal/mole')
            print('complexation energy (corrected): ' + str(float_values[4]) + ' kcal/mole')
        
        return [float(value) for value in float_values]
              
    else:
        return []