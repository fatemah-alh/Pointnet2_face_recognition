# Pointnet2_face_recognition

il codice orginario : https://github.com/charlesq34/pointnet2
Istruzione per eseguire il codice correttamente:
Questo codice è stato testato usando Cuda 10.0 ,e tensorflow 1.15
1- Controlla la compatibilità di tensorflow installata con la CUDA ,cuDNN,compiler, e la CUDA installata sul tuo serve, altrimenti non riuscerà a compilare correttamente il codice.
tabella di compatibilità: https://www.tensorflow.org/install/source#gpu
2- Esegue prima il file PreperationData.py , che dipende da Pytorch, è meglio viene installata in un ambiente separata. Controlla il path ROOT_DIR ,DATA_DIR
3- esegue i file.sh nella cartella tf_op.
4-esegue il file train_multi_gpu.py dopo aver controllato : NUM_CLASSES, e il path di dati,
5-Controlla il tipo di esperimento nel file face_dataset.py
6-per addestrare usando la normale: python train_multi_gpu.py --normal
7- Pulisci la cartella log prima ogni esperimento.


