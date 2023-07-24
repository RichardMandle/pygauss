# pygauss<br>
A collection of tools for setting up and analysing Gaussian calculations with python. <br><br>

Uses rdkit so you can start from just a smiles string, generate some 3D Coordinates and then perform a quick and dirty energy minimisation before running your gaussian calculations.<br><br>


# Functions<br>
_Minimum_Energy_Conformer_			    -	Generates the minimum energy conformer of a given molecule (as smiles) using the MMFF94 method<br>
_WriteGauss_							          -	Takes a smiles string, performs energy minimisation, then writes the 3D coordinates to a gaussian input file<br>
_MakeARCJob_							          - Creates a .sh file for launching a job on the SGE nodes of ARC3/ARC4<br>
_read_spectra_from_gaussian_output_	-	Reads spectral data (IR, Raman) from a Gaussian output file<br>
_read_complexation_energy_	  		  -	Reads the complexation energies from a gaussian calculation with counterpoise correction<br>
_extract_properties_					      -	Extracts properties from a gaussian output file (e.g. dipole, polarisability etc.)<br>
_plot_simulated_spectrum_			    	-	Plots a simulated FTIR/Raman spectrum from Gaussian data<br>
_read_energy_							          -	Reads the final energy of a molecule in a gaussian calculation<br>
_read_gaussian_route_section_    		-	Reads the route section of a gaussian file to determine what sort of job was performed<br>
_read_nmr_data_						          -	Reads NMR data from a Gaussian output file<br>
_get_nmr_correction_factors_    		-	Returns the NMR correction factors for a GAIO NMR job<br>
_do_scaling_							          -	Scales calculated NMR data according to correction factors<br>
_plot_nmr_spectrum_					        -	Plots a simulated NMR spectrum <br>
