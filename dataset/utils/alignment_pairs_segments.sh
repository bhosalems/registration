#!/bin/bash
DATAPATH="/home/csgrad/mbhosale/Datasets/CHAOS_original/CHAOS_Train_Sets/Train_Sets/MR/"
FIXED="2"
export FREESURFER_HOME=$HOME/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

for sid in $(ls "$DATAPATH")
do
	if [ $sid -ne $FIXED ]
	then
		echo "Affine transforming $DATAPATH/$sid/T1DUAL/Ground/IMG-$sid.nii.gz"
		mri_convert $DATAPATH/$sid/T1DUAL/Ground/${sid}_seg.nii.gz --apply_transform $DATAPATH/$sid/T1DUAL/DICOM_anon/InPhase/transform-${sid}.lta -o $DATAPATH/$sid/T1DUAL/Ground/${sid}_seg_intrmd.nii.gz
		mv  $DATAPATH/$sid/T1DUAL/Ground/${sid}_seg_intrmd.nii.gz  $DATAPATH/$sid/T1DUAL/Ground/${sid}_seg.nii.gz
	fi
done
