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
gen_conf.py - generates a bunch of conformers for an input structure, and creates Gaussian jobs as per the users in struction
            - to read data, use the companion conf_prop.py
'''

def initialize():
    parser = argparse.ArgumentParser(description='Generate gaussian inputs for many conformers of a structure in a .mol file (-i) or a smiles string (-smi)')
    
    # input and output file names
    parser.add_argument('-i', '--input', default='', type=str, help='Input molecule contained in a .mol filename')	
    parser.add_argument('-smi','--smiles', default='', type=str, help='Input molecule contained in a smiles (or SMARTS) string ')
    parser.add_argument('-o', '--output', default='no_name', type=str, help='name for .gjf files that we\'ll write')
    
    # Gaussian options
    parser.add_argument('-n', '--nproc', default=1, type=int, help='Number of cpu cores for gaussian jobs')
    parser.add_argument('-m', '--mem', default=1, type=int, help='ammount of RAM in GB for gaussian jobs')
    parser.add_argument('-l', '--linkoptions', default='AM1', type=str, help='Gaussian Link0 options; AM1, OPT B3LYP/6-31G(d) etc.; defaults to SPE with AM1')
    parser.add_argument('-gver', '--gauss_version', default='g16', type=str, help='Select Gaussian version (G16 or G09 at UoL)')

    # conformer generation options
    parser.add_argument('-eng', '--engine', default='ETKDGv3', type=str, help='conformer generation engine/method [ETKDGv3, ETKDGv2, ETKDG etc.]')
    parser.add_argument('-rms', '--rms', default=0.66, type=float, help='RMSD threshold for pruning identical conformers')
    parser.add_argument('-nconf', '--nconf', default=100, type=int, help='Number of conformers to generate')
    parser.add_argument('-tr', '--torsion', default=True, type=bool, help='use torsion knowledge (True/False)')
    parser.add_argument('-kn', '--knowledge', default=True, type=bool, help='Use chemical knowledge  (True/False)')
    parser.add_argument('-rc', '--random_coords', default=False, type=bool, help='use random coordinates  (True/False)')
    parser.add_argument('-rs', '--random_seed', default=61453, type=int, help='set random seed')
    parser.add_argument('-maxi', '--max_ff_iter', default=5000, type=int, help='maximum iterations in ff based minimisation')
    parser.add_argument('-mine', '--min_ff_energy', default=500, type=float, help='minimum energy in ff based minimisation')
    parser.add_argument('-ff', '--ff_variant', default='MMFF94', type=str, help='select ff variant (MMFF94, ')
    return parser
    
def get_mol_from_molfile(file_path):
    mol = Chem.MolFromMolFile(file_path)
    mol = Chem.AddHs(mol)
    return mol
    
def get_mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol

def conf_gen(mol,
             num_of_conformer=100,
             embeded_method='ETKDGv3', 
             rms_thresh = 0.675,
             use_torsion_pref = True,
             use_knowledge = True,
             use_random_coords = False,
             random_seed = 61453,
             max_iter=5000,
             min_energy_MMFF=500,
             MMFF_variant="MMFF94"):
                   
    ps = getattr(AllChem, embeded_method)()

    ps.pruneRmsThresh = rms_thresh
    ps.useExpTorsionAnglePrefs = use_torsion_pref
    ps.useBasicKnowledge = use_knowledge
    ps.useRandomCoords=use_random_coords
    ps.randomSeed = random_seed

    AllChem.EmbedMultipleConfs(mol, num_of_conformer, ps)  # creates confs
    AllChem.MMFFOptimizeMoleculeConfs(mol,maxIters=max_iter) # optimises all confs w/ ff method
    
    if num_of_conformer != mol.GetNumConformers():
        print(f'Generated {num_of_conformer} conformers using the ' + embeded_method + ' method, giving ' + str(mol.GetNumConformers()) + ' unique conformer' + ('s'*(mol.GetNumConformers()>1)))
    return mol 	
      
def write_gjf(mol, name='NoName', options='', nproc=1, vmem=1):
    os.makedirs(name, exist_ok=True)

    files = glob.glob(f"{name}/*.gjf")
    for file in os.listdir(name):
        file_path = os.path.join(name, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error occurred while deleting file {file_path}: {e}")
    
    for q in range(mol.GetNumConformers()):
        i = q+1 # python indexes from 0, sure, but SGE fails if the task array starts from 0.
        file_name = f"{name}/{name}_{i}.gjf"
        with open(file_name, 'w') as f:
            f.write(f"%chk={name}_{q}.chk\n%nprocshared={nproc}\n%mem={vmem}GB\n#p {options}\n\n{name}\n\n0 1\n")
            for i, atom in enumerate(mol.GetAtoms()):
                positions = mol.GetConformer(q).GetAtomPosition(i)
                f.write(f"{atom.GetSymbol()}\t{positions.x}\t{positions.y}\t{positions.z}\n")
            f.write('\n\n')
    return file_name

def make_sge_job(filename='noname', g_ver = 'g16', vmem=8, nproc=8, startjob=0, endjob=0):
    os.makedirs(filename, exist_ok=True)
    
    if startjob + endjob == 0: #if we've got more than 1 input file we'll make a SGE task array to submit all with a single qsub call
        Multiple = False
        
    if startjob + endjob != 0:
        Multiple = True
                
    with open(filename + '/' + filename + '.sh', 'w') as f:
        f.write('#$ -cwd \n')
        f.write('#$ -V\n')
        f.write('#$ -l h_rt=48:00:00\n')
        f.write('#$ -l h_vmem=' + str(vmem) + 'G\n')
        f.write('#$ -pe smp ' + str(nproc) + '\n')
        f.write('#$ -m be\n')
        f.write('#$ -M r.mandle@leeds.ac.uk\n')
        f.write('#$ -l disk=5G\n')
        if Multiple:
            f.write('#$ -t ' + str(startjob + 1) + '-' + str(endjob) + '\n')  # Create a task array
    
        f.write('module add gaussian\n') # TO DO - there might be a useage case for loading different versions of Gaussian, or making sure we load the **right** version here.
        f.write('export GAUSS_SCRDIR=$TMPDIR\n')
        f.write(str(g_ver + ' ' + filename + ('_$SGE_TASK_ID' * Multiple) + '.gjf\n'))
    return

def create_input():
    args = initialize().parse_args()        # parse arguments

    if args.input != '':
        mol = get_mol_from_molfile(args.input)  # load mol file as mol object

    if args.smiles != '':
        mol = get_mol_from_smiles(args.smiles)

    mol = conf_gen(mol = mol,                     # generate a bunch of conformers.
                   num_of_conformer=args.nconf,
                   embeded_method='ETKDGv3', 
                   rms_thresh = args.rms,
                   use_torsion_pref = args.torsion,
                   use_knowledge = args.knowledge,
                   use_random_coords = args.random_coords,
                   random_seed = args.random_seed,
                   max_iter=args.max_ff_iter,
                   min_energy_MMFF=args.min_ff_energy)

    write_gjf(mol,      # write them .gjf files for each conformer
            name=args.output, 
            options = args.linkoptions, 
            nproc = args.nproc, 
            vmem = args.mem)
          
    make_sge_job(filename = args.output, # write a .sh file for running them jobs
                g_ver = args.gauss_version, 
                vmem = args.mem, 
                nproc = args.nproc, 
                endjob = mol.GetNumConformers())
        
if __name__ == "__main__":
    create_input()