#include "cloth.h"
#include <Eigen/Dense>
#include <QGLWidget>
#include <Eigen/Geometry>
#include "iostream"
#include "vectormath.h"
#include "simulation.h"

using namespace std;
using namespace Eigen;

Cloth::Cloth()
{
    const std::string fileName("resources/square.obj");
    mesh_ = new Mesh(fileName);
    const Vector3d translation(5,0,4);
    mesh_->translate(translation);
    cVertPos.resize(mesh_->getNumVerts()*3);
    cVertVel.resize(mesh_->getNumVerts()*3);
    cVertPos.setZero();
    cVertVel.setZero();
    for (int idx = 0; idx < mesh_->getNumVerts(); idx++)
    {
        cVertPos.segment<3>(3*idx) = mesh_->getVert(idx);
        Vector3d zero;
        zero.setZero();
        cVertVel.segment<3>(3*idx) = zero;
    }
    cVertNormals.setZero();
    computeVertexNormals();
    computeInverseMassMatrix();
    computeStretchingData();
    computeHinges();
}

void Cloth::computeVertexNormals()
{
    cVertNormals.resize(mesh_->getNumVerts()*3);
    cVertNormals.setZero();
    cFaceNormals.clear();
    cFaceAreas.clear();
    for (int fId = 0; fId < (int) mesh_->getNumFaces(); fId++)
    {
        Vector3i fverts = mesh_->getFace(fId);
        Vector3d pts[3];
        for(int j=0; j<3; j++)
        {
            pts[j] = cVertPos.segment<3>(3*fverts[j]);
        }

        Vector3d normal = (pts[1]-pts[0]).cross(pts[2]-pts[0]);
        double norm = normal.norm();
        cFaceAreas.push_back(norm/2.0);
        cFaceNormals.push_back(normal/norm);
        for(int j=0; j<3; j++)
        {
            cVertNormals.segment<3>(3*fverts[j]) += cFaceAreas[fId]*cFaceNormals[fId];
        }
    }
    for(int i=0; i<(int)mesh_->getNumVerts(); i++)
    {
        cVertNormals.segment<3>(3*i) /= cVertNormals.segment<3>(3*i).norm();
    }
}

void Cloth::computeInverseMassMatrix()
{
    massMat.resize(mesh_->getNumVerts()*3, mesh_->getNumVerts()*3);
    massMat.setZero();
    invMassMat.resize(mesh_->getNumVerts()*3, mesh_->getNumVerts()*3);
    invMassMat.setZero();
    Matrix3d identity;
    identity.setIdentity();
    for (int fId = 0; fId < (int) mesh_->getNumFaces(); fId++)
    {
        Vector3i fverts = mesh_->getFace(fId);
        double mass = mesh_->getFaceArea(fId)*(1.0/3.0);
        for(int j=0; j<3; j++)
        {
            massMat.block<3,3>(3*fverts[j], 3*fverts[j]) += identity*mass;
            invMassMat.block<3,3>(3*fverts[j], 3*fverts[j]) += identity*(1/mass);
        }
    }
}

void Cloth::computeStretchingData()
{
    for (int fID = 0; fID < mesh_->getNumFaces(); fID++)
    {
        Vector3i points = mesh_->getFace(fID);

        Vector3d piTilda = mesh_->getVert(points[0]);
        Vector3d pjTilda = mesh_->getVert(points[1]);
        Vector3d pkTilda = mesh_->getVert(points[2]);
        Vector3d e1Tilda = pjTilda - piTilda;
        Vector3d e2Tilda = pkTilda - piTilda;

        Matrix2d gTilda;
        gTilda.coeffRef(0,0) = e1Tilda.dot(e1Tilda);
        gTilda.coeffRef(0,1) = e1Tilda.dot(e2Tilda);
        gTilda.coeffRef(1,0) = e1Tilda.dot(e2Tilda);
        gTilda.coeffRef(1,1) = e2Tilda.dot(e2Tilda);
        MatrixXd eTilda(2,3);
        eTilda.block<1,3>(0,0) = e1Tilda.transpose();
        eTilda.block<1,3>(1,0) = e2Tilda.transpose();

        MatrixXd AMat(9,4);
        MatrixXd DMat = gTilda.inverse()*eTilda;
        AMat.coeffRef(0,0) = (1.0/2.0)*DMat.coeff(0,0)*DMat.coeff(0,0);
        AMat.coeffRef(0,1) = (1.0/2.0)*DMat.coeff(1,0)*DMat.coeff(0,0);
        AMat.coeffRef(0,2) = (1.0/2.0)*DMat.coeff(0,0)*DMat.coeff(1,0);
        AMat.coeffRef(0,3) = (1.0/2.0)*DMat.coeff(1,0)*DMat.coeff(1,0);

        AMat.coeffRef(1,0) = (1.0/2.0)*DMat.coeff(0,1)*DMat.coeff(0,0);
        AMat.coeffRef(1,1) = (1.0/2.0)*DMat.coeff(1,1)*DMat.coeff(0,0);
        AMat.coeffRef(1,2) = (1.0/2.0)*DMat.coeff(0,1)*DMat.coeff(1,0);
        AMat.coeffRef(1,3) = (1.0/2.0)*DMat.coeff(1,1)*DMat.coeff(1,0);

        AMat.coeffRef(2,0) = (1.0/2.0)*DMat.coeff(0,2)*DMat.coeff(0,0);
        AMat.coeffRef(2,1) = (1.0/2.0)*DMat.coeff(1,2)*DMat.coeff(0,0);
        AMat.coeffRef(2,2) = (1.0/2.0)*DMat.coeff(0,2)*DMat.coeff(1,0);
        AMat.coeffRef(2,3) = (1.0/2.0)*DMat.coeff(1,2)*DMat.coeff(1,0);

        AMat.coeffRef(3,0) = (1.0/2.0)*DMat.coeff(0,0)*DMat.coeff(0,1);
        AMat.coeffRef(3,1) = (1.0/2.0)*DMat.coeff(1,0)*DMat.coeff(0,1);
        AMat.coeffRef(3,2) = (1.0/2.0)*DMat.coeff(0,0)*DMat.coeff(1,1);
        AMat.coeffRef(3,3) = (1.0/2.0)*DMat.coeff(1,0)*DMat.coeff(1,1);

        AMat.coeffRef(4,0) = (1.0/2.0)*DMat.coeff(0,1)*DMat.coeff(0,1);
        AMat.coeffRef(4,1) = (1.0/2.0)*DMat.coeff(1,1)*DMat.coeff(0,1);
        AMat.coeffRef(4,2) = (1.0/2.0)*DMat.coeff(0,1)*DMat.coeff(1,1);
        AMat.coeffRef(4,3) = (1.0/2.0)*DMat.coeff(1,1)*DMat.coeff(1,1);

        AMat.coeffRef(5,0) = (1.0/2.0)*DMat.coeff(0,2)*DMat.coeff(0,1);
        AMat.coeffRef(5,1) = (1.0/2.0)*DMat.coeff(1,2)*DMat.coeff(0,1);
        AMat.coeffRef(5,2) = (1.0/2.0)*DMat.coeff(0,2)*DMat.coeff(1,1);
        AMat.coeffRef(5,3) = (1.0/2.0)*DMat.coeff(1,2)*DMat.coeff(1,1);

        AMat.coeffRef(6,0) = (1.0/2.0)*DMat.coeff(0,0)*DMat.coeff(0,2);
        AMat.coeffRef(6,1) = (1.0/2.0)*DMat.coeff(1,0)*DMat.coeff(0,2);
        AMat.coeffRef(6,2) = (1.0/2.0)*DMat.coeff(0,0)*DMat.coeff(1,2);
        AMat.coeffRef(6,3) = (1.0/2.0)*DMat.coeff(1,0)*DMat.coeff(1,2);

        AMat.coeffRef(7,0) = (1.0/2.0)*DMat.coeff(0,1)*DMat.coeff(0,2);
        AMat.coeffRef(7,1) = (1.0/2.0)*DMat.coeff(1,1)*DMat.coeff(0,2);
        AMat.coeffRef(7,2) = (1.0/2.0)*DMat.coeff(0,1)*DMat.coeff(1,2);
        AMat.coeffRef(7,3) = (1.0/2.0)*DMat.coeff(1,1)*DMat.coeff(1,2);

        AMat.coeffRef(8,0) = (1.0/2.0)*DMat.coeff(0,2)*DMat.coeff(0,2);
        AMat.coeffRef(8,1) = (1.0/2.0)*DMat.coeff(1,2)*DMat.coeff(0,2);
        AMat.coeffRef(8,2) = (1.0/2.0)*DMat.coeff(0,2)*DMat.coeff(1,2);
        AMat.coeffRef(8,3) = (1.0/2.0)*DMat.coeff(1,2)*DMat.coeff(1,2);

        MatrixXd CMat(6,9);
        CMat.setZero();
        Matrix3d identity;
        identity.setIdentity();
        CMat.block<3,3>(0,0) = identity*-1;
        CMat.block<3,3>(0,3) = identity;
        CMat.block<3,3>(3,0) = identity*-1;
        CMat.block<3,3>(3,6) = identity;

        faceEs.push_back(eTilda);
        faceGs.push_back(gTilda);
        faceCs.push_back(CMat);
        faceDs.push_back(DMat);
        faceAs.push_back(AMat);
    }
}

void Cloth::computeHinges()
{
    hinges.clear();
    for (int f0ID = 0; f0ID < mesh_->getNumFaces(); f0ID++)
    {
        Vector3i f0Points = mesh_->getFace(f0ID);
        for (int f1ID = f0ID+1; f1ID < mesh_->getNumFaces(); f1ID++)
        {
            Vector3i f1Points = mesh_->getFace(f1ID);
            vector<int> commonPoints0;
            vector<int> commonPoints1;
            commonPoints0.clear();
            commonPoints1.clear();
            int notCommon0;
            for (int point = 0; point < 3; point++)
            {
                if (f0Points[point] == f1Points[0])
                {
                    commonPoints0.push_back(point);
                    commonPoints1.push_back(0);
                }
                else if (f0Points[point] == f1Points[1])
                {
                    commonPoints0.push_back(point);
                    commonPoints1.push_back(1);
                }
                else if (f0Points[point] == f1Points[2])
                {
                    commonPoints0.push_back(point);
                    commonPoints1.push_back(2);
                }
                else
                {
                    notCommon0 = point;
                }
            }
            if (commonPoints0.size() == 2)
            {
                int pI = f0Points[commonPoints0[0]];
                int pJ = f0Points[commonPoints0[1]];
                int f0 = f0ID;
                int f1 = f1ID;
                int kLonerF0 = f0Points[notCommon0];
                int lLonerF1;
                double length = (mesh_->getVert(pI) - mesh_->getVert(pJ)).norm();
                double f0rArea = mesh_->getFaceArea(f0);
                double f1rArea = mesh_->getFaceArea(f1);
                for (int i = 0; i<3; i++)
                {
                    if (f1Points[i] != f0Points[0] && f1Points[i] != f0Points[1] && f1Points[i] != f0Points[2])
                    {
                        lLonerF1 = f1Points[i];
                    }
                }
                hinges.push_back(Hinge(pI, pJ, f0, f1, kLonerF0, lLonerF1, length, f0rArea, f1rArea));
            }
        }
    }
//    cout<<"Hinges:"<<hinges.size()<<endl;
}


void Cloth::render()
{
    computeVertexNormals();
    glShadeModel(GL_SMOOTH);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
    glEnable ( GL_COLOR_MATERIAL );
    Vector3d color(0.7, 0, 0.7);
    glColor4d(color[0], color[1], color[2], 1.0);

    glPushMatrix();
    {
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);

        glVertexPointer(3, GL_DOUBLE, 0, getCurrentVertexPointer());
        glNormalPointer(GL_DOUBLE, 0, getCurrentVertexNormalsPointer());

        glDrawElements(GL_TRIANGLES, mesh_->getNumFaces()*3, GL_UNSIGNED_INT, getCurrentFacePointer());

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
    }
    glPopMatrix();

}
