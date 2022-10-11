#!/bin/bash
OUTPUTPATH="/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR"
DATAPATH="/data_local/mbhosale/CHAOS_bkp/CHAOS_Train_Sets/Train_Sets/MR"
for sid in $(ls "$DATAPATH/T2SPIR/")
do
	arrIN=(${sid//_/ })
	IN=${arrIN[1]}
	arrIN=(${IN//./ })
	IN=${arrIN[0]}
	if [[ $sid == *"label"* ]]; then
		cp "$DATAPATH/T2SPIR/$sid" "$OUTPUTPATH/$IN/T2SPIR/Ground/seg-$IN.nii.gz"
	#else
	#	cp "$DATAPATH/T2SPIR/$sid" "$OUTPUTPATH/$IN/T2SPIR/DICOM_anon/IMG-$IN.nii.gz"
	fi
done

for sid in $(ls "$DATAPATH/T1DUAL/InPhase/")
do
	arrIN=(${sid//_/ })
	IN=${arrIN[1]}
	arrIN=(${IN//./ })
	IN=${arrIN[0]}
	if [[ $sid == *"label"* ]]; then
		cp "$DATAPATH/T1DUAL/InPhase/$sid" "$OUTPUTPATH/$IN/T1DUAL/Ground/seg-$IN.nii.gz"
	#else
	#	cp "$DATAPATH/T1DUAL/InPhase/$sid" "$OUTPUTPATH/$IN/T1DUAL/DICOM_anon/InPhase/IMG-$IN.nii.gz"
	fi
done

#for sid in $(ls "$DATAPATH/T1DUAL/OutPhase/")
#do
#	arrIN=(${sid//_/ })
#	IN=${arrIN[1]}
#	arrIN=(${IN//./ })
#	IN=${arrIN[0]}
#	if [[ $sid != *"label"* ]]; then
#		cp "$DATAPATH/T1DUAL/OutPhase/$sid" "$OUTPUTPATH/$IN/T1DUAL/DICOM_anon/OutPhase/IMG-$IN.nii.gz"
#	fi
#done
