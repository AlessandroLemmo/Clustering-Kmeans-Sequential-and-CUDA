# Clustering-Kmeans-Sequential-and-CUDA

The application implements the K-Means clustering with two modalities: sequential and parallel in CUDA that is an extension of C programming with functionalities for the parallel code implementation, developed by Nvidia.

The times of exexution of the two modalities have been compared considering the following aspects:
- variation of number of threads for the parallel mode
- variation of number of point to clustering
- variation of number of clusters

The project was developed with the operating system Windows an the IDE Visual Studio. For replicate the result download the project, open Visual Studio and compute the following steps:
1. Click on File -> Open -> Project/Solution
2. Select the .sln project file and click to Open
3. Configuration Manager -> Active solution configuration -> Release
4. Configuration Manager -> Active solution platform -> x64
5. Build Dependencies -> Build Customizations -> Tick on CUDA
6. Properties -> Tab CUDA C/C++ -> Target Machine Platform -> 64-bit
7. Properties -> Tab Linker -> Sub-Tab Input -> Additional Dependencies -> cudart.lib 

In the folder _report_ there are:
- Relation of the project
- File excel that reports all time results
- Slides of presentation
