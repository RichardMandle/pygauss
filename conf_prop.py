# ! /usr/bin/env python

import argparse
import glob
import os
import platform

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np
import re

'''
conf_prop.py - Reads gaussian output files for a bunch of conformers and obtained probability weighted properties.
			 - to generate conformational things, use the gen_conf.py script.
'''

def initialize():
    parser = argparse.ArgumentParser(description='create a functional .gjf file for RESP charges from a .mol file')
    
    # input and output file names
    parser.add_argument('-t', '--temperature', default=298, type=float, help='temperature for probability calculation (float)')	
    parser.add_argument('-i', '--path', default='', type=str, help='path of directory to retrieve Gaussian output files from (defaults to cwd/pwd)')
    parser.add_argument('-e', '--extension', default='.log', type=str, help='file extension of Gaussian output files (usually .log or .out!)')	
    
    return parser
    
def find_output_files(directory, extension):
    if not extension.startswith('.'):
        extension = '.' + extension

    output_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                # Append the file path to the list
                output_files.append(os.path.join(root, file))

    return output_files

def read_energy(file):
    with open(file) as f:
        return [float(line.split(' = ')[1].split(' A.U.')[0]) for line in f if 'SCF Done:' in line]
   
def calc_prob(energy, temperature = 298):
    Kb = 0.008314 # Boltzmann's constant
    delta_e_kj = (energy - np.min(energy))* 4.184
    probability = np.exp(-delta_e_kj / (Kb * temperature )) / np.sum(np.exp(-delta_e_kj / (Kb * temperature )))
    
    return delta_e_kj, probability 
	
def extract_properties(file_path):
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
            properties["Dipole tensor"] = np.abs(dipole_values) # ooh thats not a good idea (np.abs!)
            properties["Dipole moment"] = np.sqrt(np.sum(dipole_values**2))
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
    
def calculate_weighted_average(properties_list, probabilities):
    weighted_averages = {}     # Initialize acca for each property
    for properties, probability in zip(properties_list, probabilities):
        for key, value in properties.items():
            if key not in weighted_averages:
                if isinstance(value, list):
                    weighted_averages[key] = [0.0] * len(value)
                else: 
                    weighted_averages[key] = np.zeros_like(value)

            weighted_averages[key] += np.array(value) * probability

    total_probability = sum(probabilities)
    for key in weighted_averages.keys():
        weighted_averages[key] /= total_probability
    
    print('Probability averaged properties:')
    print(weighted_averages)
    return weighted_averages

def read_output():
    args = initialize().parse_args()        # parse arguments
    args.path = os.path.join(os.getcwd(), args.path) 
    
    file_list = find_output_files(args.path, args.extension)
    
    properties_list = []
    energy_list = []
    
    for file in file_list:
        energy_list.append(read_energy(file)[-1])
        properties_list.append(extract_properties(file))
        
    delta_E, prob = calc_prob(energy_list,args.temperature) # get probabilities at temperature for energies returned
    
    return delta_E, prob, properties_list
    
if __name__ == "__main__":
    delta_E, probaibilities, properties = read_output() # read the output files
    calculate_weighted_average(properties, probaibilities) #get the weighted average of the properties.
    