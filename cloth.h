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

    Eigen::VectorXd cVertPos;
    Eigen::VectorXd cVertVel;
    Eigen::VectorXd cVertNormals;
    Eigen::MatrixXd massMat;
    Eigen::MatrixXd invMassMat;
    std::vector<Eigen::Vector3d> cFaceNormals;
    std::vector<double> cFaceAreas;

    Eigen::MatrixXd getETildaMatrix(int faceId) {return faceEs[faceId];}
    Eigen::Matrix2d getGTildaMatrix(int faceId) {return faceGs[faceId];}
    Eigen::MatrixXd getCMatrix(int faceId) {return faceCs[faceId];}
    Eigen::MatrixXd getAMatrix(int faceId) {return faceAs[faceId];}
    Eigen::MatrixXd getDMatrix(int faceId) {return faceDs[faceId];}

private:

    void computeStretchingData();

    std::vector<Eigen::MatrixXd> faceEs;
    std::vector<Eigen::Matrix2d> faceGs;
    std::vector<Eigen::MatrixXd> faceCs;
    std::vector<Eigen::MatrixXd> faceAs;
    std::vector<Eigen::MatrixXd> faceDs;
    Mesh *mesh_;
};

#endif // CLOTH_H
