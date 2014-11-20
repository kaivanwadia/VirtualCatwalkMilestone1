#include "cloth.h"
#include <Eigen/Dense>
#include <QGLWidget>
#include <Eigen/Geometry>
#include "iostream"
#include "vectormath.h"

Cloth::Cloth()
{
    const std::string fileName("resources/square.obj");
    mesh_ = new Mesh(fileName);
    const Eigen::Vector3d translation(5,0,4);
    mesh_->translate(translation);
    cVertPos.resize(mesh_->getNumVerts()*3);
    cVertVel.resize(mesh_->getNumVerts()*3);
    cVertPos.setZero();
    cVertVel.setZero();
    for (int idx = 0; idx < mesh_->getNumVerts(); idx++)
    {
        cVertPos.segment<3>(3*idx) = mesh_->getVert(idx);
        Eigen::Vector3d zero;
        zero.setZero();
        cVertVel.segment<3>(3*idx) = zero;
    }
    cVertNormals.setZero();
    computeVertexNormals();
}

void Cloth::computeVertexNormals()
{
    cVertNormals.resize(mesh_->getNumVerts()*3);
    cVertNormals.setZero();
    cFaceNormals.clear();
    cFaceAreas.clear();
    for (int fId = 0; fId < (int) mesh_->getNumFaces()/3; fId++)
    {
        Eigen::Vector3i fverts = mesh_->getFace(fId);
        Eigen::Vector3d pts[3];
        for(int j=0; j<3; j++)
        {
            pts[j] = cVertPos.segment<3>(3*fverts[j]);
        }

        Eigen::Vector3d normal = (pts[1]-pts[0]).cross(pts[2]-pts[0]);
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

void Cloth::render()
{
//    Eigen::Matrix3d rot = VectorMath::rotationMatrix(theta);

    glShadeModel(GL_SMOOTH);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
    glEnable ( GL_COLOR_MATERIAL );
    Eigen::Vector3d color(0.7, 0, 0.7);
    glColor4d(color[0], color[1], color[2], 1.0);

    glPushMatrix();
    {
//        GLdouble xform[16];
//        for(int i=0; i<3; i++)
//        {
//            for(int j=0; j<3; j++)
//                xform[4*j+i] = rot.coeff(i,j);
//            xform[4*i+3] = 0;
//            xform[12+i] = c[i];
//        }
//        xform[15] = 1.0;
//        glMultMatrixd(xform);
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
