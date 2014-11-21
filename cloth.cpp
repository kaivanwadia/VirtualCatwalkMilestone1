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
        double mass = cFaceAreas[fId]*(1.0/3.0);
        for(int j=0; j<3; j++)
        {
            massMat.block<3,3>(3*fverts[j], 3*fverts[j]) += identity*mass;
            invMassMat.block<3,3>(3*fverts[j], 3*fverts[j]) += identity*(1/mass);
        }
    }
}

void Cloth::render()
{
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
