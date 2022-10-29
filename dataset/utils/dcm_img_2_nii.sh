#!/bin/bash
# Convert dicom-like images to nii files in 3D
# This is the first step for image pre-processing

# Feed path to the downloaded data here
DATAPATH="/home/csgrad/mbhosale/Datasets/CHAOS/CHAOS_Train_Sets/Train_Sets/MR" # please put chaos dataset training fold here which contains ground truth
OUTPATH="/home/csgrad/mbhosale/Datasets/CHAOS/CHAOS_Train_Sets/Train_Sets/MR" 
for sid in $(ls "$DATAPATH")
do
	dcm2nii "$DATAPATH/$sid/T2SPIR/DICOM_anon";
	find "$DATAPATH/$sid/T2SPIR" -name "*.nii.gz" -exec mv {} "$OUTPATH/$sid/T2SPIR/DICOM_anon/IMG-$sid.nii.gz" \;
	dcm2nii "$DATAPATH/$sid/T1DUAL/DICOM_anon/InPhase";
	find "$DATAPATH/$sid/T1DUAL/DICOM_anon/InPhase" -name "*.nii.gz" -exec mv {} "$OUTPATH/$sid/T1DUAL/DICOM_anon/InPhase/IMG-$sid.nii.gz" \;
	dcm2nii "$DATAPATH/$sid/T1DUAL/DICOM_anon/OutPhase";
	find "$DATAPATH/$sid/T1DUAL/DICOM_anon/OutPhase" -name "*.nii.gz" -exec mv {} "$OUTPATH/$sid/T1DUAL/DICOM_anon/OutPhase/IMG-$sid.nii.gz" \;
done;
