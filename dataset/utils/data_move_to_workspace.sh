#!/bin/bash
DATAPATH="/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR"
OUTPUTPATH="/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR_workspace"
mkdir -p "$OUTPUTPATH/T2SPIR"
for sid in $(ls "$DATAPATH")
do
	echo $sid
	find "$DATAPATH/$sid/T2SPIR/DICOM_anon" -name "*.nii.gz" #-exec mv {} "$OUTPUTPATH/T2SPIR/image_$sid.nii.gz" \;
	#find "$DATAPATH/$sid/T2SPIR/Ground" -name "*.nii.gz" -exec mv {} "$OUTPUTPATH/T2SPIR/label_$sid.nii.gz" \;
done
#mkdir -p "$OUTPUTPATH/T1DUAL/InPhase"
#for sid in $(ls "$DATAPATH")
#do
#	find "$DATAPATH/$sid/T1DUAL/DICOM_anon/InPhase" -name "*.nii.gz" -exec mv {} "$OUTPUTPATH/T1DUAL/InPhase/image_$sid.nii.gz" \;
#	find "$DATAPATH/$sid/T1DUAL/Ground" -name "*.nii.gz" -exec mv {} "$OUTPUTPATH/T1DUAL/InPhase/label_$sid.nii.gz" \;
#
#done
#mkdir -p "$OUTPUTPATH/T1DUAL/OutPhase"
#for sid in $(ls "$DATAPATH")
#do
#	find "$DATAPATH/$sid/T1DUAL/DICOM_anon/OutPhase" -name "*.nii.gz" -exec mv {} "$OUTPUTPATH/T1DUAL/OutPhase/image_$sid.nii.gz" \;
#	find "$DATAPATH/$sid/T1DUAL/Ground" -name "*.nii.gz" -exec mv {} "$OUTPUTPATH/T1DUAL/OutPhase/label_$sid.nii.gz" \;
#done

