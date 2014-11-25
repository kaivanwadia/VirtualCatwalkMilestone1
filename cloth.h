#ifndef CLOTH_H
#define CLOTH_H
#include <QGLWidget>
#include<mesh.h>
#include<Eigen/Core>
#include <QMutex>
#include <Eigen/Sparse>
#include <vector>
#include <set>
#include <QMutex>




struct Hinge
{
public:
    Hinge(int pI, int pJ, int f0, int f1, int kLonerF0, int lLonerF1, double length, double f0rArea, double f1rArea) : pI(pI), pJ(pJ), f0(f0), f1(f1), kLonerF0(kLonerF0), lLonerF1(lLonerF1), rLength(length), f0rArea(f0rArea), f1rArea(f1rArea)
    {
    }
    int pI;
    int pJ;
    int f0;
    int f1;
    int kLonerF0;
    int lLonerF1;
    double rLength;
    double f0rArea;
    double f1rArea;
};

class Cloth
{
public:
    Cloth();
    void render();
    void computeVertexNormals();
    void computeInverseMassMatrix();
    void computeHinges();
    void loadTexture();

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
    std::vector<Hinge> hinges;

    Eigen::MatrixXd getETildaMatrix(int faceId) {return faceEs[faceId];}
    Eigen::Matrix2d getGTildaMatrix(int faceId) {return faceGs[faceId];}
    Eigen::MatrixXd getCMatrix(int faceId) {return faceCs[faceId];}
    Eigen::MatrixXd getAMatrix(int faceId) {return faceAs[faceId];}
    Eigen::MatrixXd getDMatrix(int faceId) {return faceDs[faceId];}

private:

    void computeStretchingData();
    GLuint clothTex;
    std::vector<Eigen::MatrixXd> faceEs;
    std::vector<Eigen::Matrix2d> faceGs;
    std::vector<Eigen::MatrixXd> faceCs;
    std::vector<Eigen::MatrixXd> faceAs;
    std::vector<Eigen::MatrixXd> faceDs;
    Mesh *mesh_;
};

#endif // CLOTH_H
