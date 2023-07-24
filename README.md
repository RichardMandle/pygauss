# pygauss<br>
A collection of tools for setting up and analysing Gaussian calculations with python. <br><br>

Uses rdkit so you can start from just a smiles string, generate some 3D Coordinates and then perform a quick and dirty energy minimisation before running your gaussian calculations.<br><br>


# Functions<br>
Minimum_Energy_Conformer			    -	Generates the minimum energy conformer of a given molecule (as smiles) using the MMFF94 method<br>
WriteGauss							          -	Takes a smiles string, performs energy minimisation, then writes the 3D coordinates to a gaussian input file<br>
MakeARCJob							          - Creates a .sh file for launching a job on the SGE nodes of ARC3/ARC4<br>
read_spectra_from_gaussian_output	-	Reads spectral data (IR, Raman) from a Gaussian output file<br>
read_complexation_energy	  		  -	Reads the complexation energies from a gaussian calculation with counterpoise correction<br>
extract_properties					      -	Extracts properties from a gaussian output file (e.g. dipole, polarisability etc.)<br>
plot_simulated_spectrum			    	-	Plots a simulated FTIR/Raman spectrum from Gaussian data<br>
read_energy							          -	Reads the final energy of a molecule in a gaussian calculation<br>
read_gaussian_route_section    		-	Reads the route section of a gaussian file to determine what sort of job was performed<br>
read_nmr_data						          -	Reads NMR data from a Gaussian output file<br>
get_nmr_correction_factors    		-	Returns the NMR correction factors for a GAIO NMR job<br>
do_scaling							          -	Scales calculated NMR data according to correction factors<br>
plot_nmr_spectrum					        -	Plots a simulated NMR spectrum <br>
