#ifndef CLOTH_H
#define CLOTH_H

#include<mesh.h>
#include<Eigen/Core>

class Cloth
{
public:
    Cloth();
    void render();
    void computeVertexNormals();
    void computeInverseMassMatrix();

    const Mesh &getMesh() const {return *mesh_;}
    const int *getCurrentFacePointer() const {return getMesh().getFacePointer();}
    const double *getCurrentVertexPointer() const {return &cVertPos[0];}
    const double *getCurrentVertexNormalsPointer() const {return &cVertNormals[0];}

private:
    Eigen::VectorXd cVertPos;
    Eigen::VectorXd cVertVel;
    Eigen::VectorXd cVertNormals;
    Eigen::MatrixXd massMat;
    Eigen::MatrixXd invMassMat;
    std::vector<Eigen::Vector3d> cFaceNormals;
    std::vector<double> cFaceAreas;

    Mesh *mesh_;
};

#endif // CLOTH_H
