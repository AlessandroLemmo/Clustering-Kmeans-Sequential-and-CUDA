# Clustering-Kmeans-Sequential-and-CUDA

The application implements the K-Means clustering with two modalities: sequential and parallel in CUDA that is an extension of C programming with functionalities for the parallel code implementation, developed by Nvidia.

The times of exexution of the two modalities have been compared considering the following aspects:
- variation of number of threads for the parallel mode
- variation of number of point to clustering
- variation of number of clusters

The project was developed with Visual Studio. For replicate the result download the project, open Visual Studio and compute the following steps:
_ Click on File -> Open -> Project/Solution
- Select the .sln project file and click to Open
- Configuration Manager -> Active solution configuration -> Release
- Configuration Manager -> Active solution platform -> x64
- Build Dependencies -> Build Customizations -> Tick on CUDA
- Properties -> Tab CUDA C/C++ -> Target Machine Platform -> 64-bit
- Properties -> Tab Linker -> Sub-Tab Input -> Additional Dependencies -> cudart.lib 

In the folder _report_ there are:
- Relation of the project
- File excel that reports all time results
- Powerpoint presentation of the project
