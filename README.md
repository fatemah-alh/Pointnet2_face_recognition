# Pointnet2_face_recognition

Using private data set of 3d point clouds of faces, annotated with (face id, expression on face, and Unicode of face).
we trained a neural network (PointNet++) for three main tasks: 
face recognition, expression recognition, and Unicode recognition.
as a results, we obtained an accuracy of 94.3% and loss of 0.22 on the face recognition, 64% accuracy and loss of 0.40 on expression, and worse results on Unicode 56%accuracy and loss of 0.48.
These different results are due to the variety of the distribution of data on classes in each of the above tasks. 
This project was done for my exam "Computer graphics and 3D"
