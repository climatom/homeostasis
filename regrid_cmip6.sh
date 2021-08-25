#!/bin/bash

set -e

regrid="True"

# Directory holding the files
di="/home/lunet/gytm3/Homeostasis/CMIP6/Merged/"

# Grid we're aiming for
grid="/home/lunet/gytm3/Homeostasis/CMIP6/target_grid.txt"

# Output for regridded
do="${di}Regridded/"

# Mask file
mask="/home/lunet/gytm3/Homeostasis/CMIP6/mask.nc"

# Iterate over the files, regrid, and then scatter to reduce file size. 
di="/home/lunet/gytm3/Homeostasis/CMIP6/Merged/"

if [ "${regrid}" = "True" ]; then

	for f in ${di}*nc; do
		# Tw
		oname=${f/${di}/${do}}
		cmd="cdo -O -s mul ${mask} -remapbil,${grid} ${f} ${oname}"
		$cmd	
		echo; echo "Processed ${f}"; echo 
	done
fi

