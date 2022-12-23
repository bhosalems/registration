#!/bin/bash
DATAPATH="/home/csgrad/mbhosale/Datasets/CHAOS_preprocessed/CHAOS_Train_Sets/Train_Sets/MR/"
FIXED="33"
export FREESURFER_HOME=$HOME/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

for sid in $(ls "$DATAPATH")
do
	if [ $sid -ne $FIXED ]
	then
		echo "Affine transforming $DATAPATH/$sid/T1DUAL/DICOM_anon/InPhase/IMG-$sid.nii.gz"
		mri_robust_register --mov $DATAPATH/$sid/T1DUAL/DICOM_anon/InPhase/IMG-$sid.nii.gz --dst $DATAPATH/$FIXED/T1DUAL/DICOM_anon/InPhase/IMG-$FIXED.nii.gz --lta $DATAPATH/$sid/T1DUAL/DICOM_anon/InPhase/transform-${sid}.lta --mapmov $DATAPATH/$sid/T1DUAL/DICOM_anon/InPhase/IMG-${sid}to${FIXED}.nii.gz --iscale --satit --affine
		mv  $DATAPATH/$sid/T1DUAL/DICOM_anon/InPhase/IMG-${sid}to${FIXED}.nii.gz  $DATAPATH/$sid/T1DUAL/DICOM_anon/InPhase/IMG-${sid}.nii.gz
	fi
done
