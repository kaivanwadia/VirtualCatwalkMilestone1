VirtualCatwalkMilestone1
=================================
Kaivan and Vismay
--------------------------------

Designed and implemented a system to simulate the behaiour of cloth in contact with a rigid body. The cloth was simulated as a triangular mesh with each triangular face of the mesh being controlled by a stretching and bending force.

Compile the code:
-Open up catwalk.pro in QT and press Build.

Special Instruction: 
-To get stretching working, divide the default Stretching K by 10.
-Since you have to divide StretchingK by 10, you should also divide BendingK by 10 so that the forces are consistent otherwise bending force will be greater than stretching force.

Things that dont work properly:
-Stretching breaks on collision with the default settings (k = 10000)
-2 by 4 Rotation (7 and 9 keys) makes the object disappear almost always. Haven't edited any of the default code provided. We looked through it but couldn't figure out why it was breaking.





