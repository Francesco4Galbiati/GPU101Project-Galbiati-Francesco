# GPU101Project-Galbiati-Francesco
Repository per il progetto di GPU101, edizione autunno 2022

Guida alla compilazione e all'esecuzione del codice:

Installazione:
 1. se si dispone di una GPU Nvidia occorre o avere un sistema operativo Linux o installare wsl2 su Windows (guida per wsl2: https://learn.microsoft.com/en-us/windows/wsl/install), una volta a disposizione di un sistema Linux serve installare CUDA e i driver corrispondenti (guida all'installazione: https://developer.nvidia.com/cuda-downloads)

 2. se non di dispone di una GPU Nvidia occorre usare Google Colab (guida all'uso di colab e di xterm: https://github.com/albertozeni/gpu_course_colab)

Compilazione: una volta aperto il terminale Linux, wsl2 o xterm su Colab si utilizzerà nvcc per la compilazione del codice (se si sta usando Colab occorre prima caricarvi il file .cu):

 1. posizionarsi nella directory del file da compilare
 2. digitare "nvcc main.cu -O3 -o symgs_smoother"
 3. se il compilatore non restituisce messaggi di errore la compilazione è andata a buon fine

Esecuzione: una volta in possesso del file eseguibile occorre:
 
 1. caricare il file di input disponibile al link: https://www.dropbox.com/s/jzn573j0z9ffl7h/kmer_V4a.mtx?dl=0 (essendo un file molto pesante conviene caricarlo inizialmente sul proprio Google Drive e successivamente collegare il Drive al notebook Colab che si sta utilizzando)
 2. digitare "./symgs_smoother kmer_V4a.mtx" e aspettare il termine dell'esecuzione
