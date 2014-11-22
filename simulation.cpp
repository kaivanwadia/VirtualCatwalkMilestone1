#include "simulation.h"
#include <QGLWidget>
#include "simparameters.h"
#include <iostream>
#include <Eigen/Geometry>
#include <QDebug>
#include "SOIL.h"
#include "rigidbodytemplate.h"
#include "rigidbodyinstance.h"
#include "cloth.h"
#include "vectormath.h"
#include <Eigen/Dense>
#include "mesh.h"
#include "signeddistancefield.h"

const double PI = 3.1415926535898;

using namespace Eigen;
using namespace std;

Simulation::Simulation(const SimParameters &params) : params_(params), time_(0), floorTex_(0)
{
    loadRigidBodies();
    bodyInstance_ = NULL;
    cloth_ = NULL;
    clearScene();
}

Simulation::~Simulation()
{
    delete bodyInstance_;
    delete bodyTemplate_;
    delete cloth_;
}

void Simulation::initializeGL()
{
    loadFloorTexture();
}

void Simulation::loadRigidBodies()
{
    string objname("resources/2by4.obj");
    bodyTemplate_ = new RigidBodyTemplate(objname);
    string sdfname("resources/2by4.sdf");
    bodyTemplate_->computeSDF(sdfname.c_str());
}

void Simulation::loadFloorTexture()
{
    floorTex_ = SOIL_load_OGL_texture("resources/grid.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_INVERT_Y |  SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT | SOIL_FLAG_MIPMAPS);
    if(floorTex_ != 0)
    {
        glBindTexture(GL_TEXTURE_2D, floorTex_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    }
}


void Simulation::renderFloor()
{
    renderLock_.lock();

    glColor4f(1.0, 1.0, 1.0, 1.0);

    if(floorTex_)
    {
        glBindTexture(GL_TEXTURE_2D, floorTex_);
        glEnable(GL_TEXTURE_2D);
    }
    else
    {
        glColor3f(0.5, 0.5, 0.5);
        glDisable(GL_TEXTURE_2D);
    }

    double texsize = 5.0;
    double gridsize = 1000.0;

    double texmax = gridsize/texsize;

    Vector3d tangent1(1.0,0,0);
    Vector3d tangent2(0,-1.0,0);

    Vector3d corner;

    glBegin(GL_QUADS);
    {
        glTexCoord2f(texmax, texmax);
        glNormal3f(0, 0, 1.0);
        corner = -gridsize*tangent1 + gridsize*tangent2;
        glVertex3f(corner[0], corner[1], corner[2]);

        glTexCoord2f(texmax, -texmax);
        glNormal3f(0, 0, 1.0);
        corner = -gridsize*tangent1 - gridsize*tangent2;
        glVertex3f(corner[0], corner[1], corner[2]);

        glTexCoord2f(-texmax, -texmax);
        glNormal3f(0, 0, 1.0);
        corner = gridsize*tangent1 - gridsize*tangent2;
        glVertex3f(corner[0], corner[1], corner[2]);

        glTexCoord2f(-texmax, texmax);
        glNormal3f(0, 0, 1.0);
        corner = gridsize*tangent1 + gridsize*tangent2;
        glVertex3f(corner[0], corner[1], corner[2]);
    }
    glDisable(GL_TEXTURE_2D);
    glEnd();
    renderLock_.unlock();
}

void Simulation::renderObjects()
{
    renderLock_.lock();
    {
        bodyInstance_->render();
        cloth_->render();
    }
    renderLock_.unlock();
}

void Simulation::takeSimulationStep()
{
    time_ += params_.timeStep;

    bodyInstance_->c += params_.timeStep*bodyInstance_->cvel;
    bodyInstance_->theta = VectorMath::axisAngle(VectorMath::rotationMatrix(params_.timeStep*bodyInstance_->w)*VectorMath::rotationMatrix(bodyInstance_->theta));

    //Do velocity verlet
    cloth_->cVertPos = cloth_->cVertPos + params_.timeStep*cloth_->cVertVel;
    VectorXd forces = computeForces();
    cout<<"Forces:\n"<<forces.segment<3>(0)<<endl;
    cloth_->cVertVel = cloth_->cVertVel + params_.timeStep*cloth_->invMassMat*forces;
    if (params_.pinCorner)
    {
        cloth_->cVertVel.segment<3>(0)<<0,0,0;
    }
}

VectorXd Simulation::computeForces()
{
    VectorXd forces(cloth_->getMesh().getNumVerts()*3);
    forces.setZero();
    if (params_.activeForces & SimParameters::F_GRAVITY)
    {
        forces = forces + computeGravity();
    }
    if (params_.activeForces & SimParameters::F_STRETCHING)
    {
        forces = forces + computeStretchingForce();
    }
    return forces;
}

VectorXd Simulation::computeStretchingForce()
{
    VectorXd stForce(cloth_->getMesh().getNumVerts()*3);
    stForce.setZero();
    VectorXd forceLocal(9);
    forceLocal.setZero();
    cout<<"\nTIMESTEP ---------------:\n"<<endl;
    for (int fID = 0; fID < cloth_->getMesh().getNumFaces(); fID++)
    {
        Vector3i points = cloth_->getMesh().getFace(fID);
        Vector3d pi = cloth_->cVertPos.segment<3>(points[0]*3);
        Vector3d pj = cloth_->cVertPos.segment<3>(points[1]*3);
        Vector3d pk = cloth_->cVertPos.segment<3>(points[2]*3);
        Vector3d e1 = pj - pi;
        Vector3d e2 = pk - pi;
        Matrix2d g;
        g.coeffRef(0,0) = e1.dot(e1);
        g.coeffRef(0,1) = e1.dot(e2);
        g.coeffRef(1,0) = e1.dot(e2);
        g.coeffRef(1,1) = e2.dot(e2);
        Matrix2d deltaG = g - cloth_->getGTildaMatrix(fID);
        MatrixXd bMatrix(4,6);
        bMatrix.setZero();
        bMatrix.block<1,3>(0,0) = 2*e1.transpose();
        bMatrix.block<1,3>(1,0) = e2.transpose();
        bMatrix.block<1,3>(1,3) = e1.transpose();
        bMatrix.block<1,3>(2,0) = e2.transpose();
        bMatrix.block<1,3>(2,3) = e1.transpose();
        bMatrix.block<1,3>(3,3) = 2*e2.transpose();
        MatrixXd dMatrix = cloth_->getDMatrix(fID);
//        Matrix3d epsilon = (dMatrix.transpose() * deltaG * dMatrix)/2.0;
        Matrix3d epsilon = (cloth_->getETildaMatrix(fID).transpose()*cloth_->getGTildaMatrix(fID).inverse() * deltaG * cloth_->getGTildaMatrix(fID).inverse()*cloth_->getETildaMatrix(fID))/2.0;
        VectorXd epsilonV(9);
        epsilonV.segment<3>(0) = epsilon.block<1,3>(0,0).transpose();
        epsilonV.segment<3>(3) = epsilon.block<1,3>(1,0).transpose();
        epsilonV.segment<3>(6) = epsilon.block<1,3>(2,0).transpose();
        MatrixXd cMatrix = cloth_->getCMatrix(fID);
        MatrixXd aMatrix = cloth_->getAMatrix(fID);
        forceLocal = (-2.0)*params_.stretchingK*cloth_->getMesh().getFaceArea(fID)
                *cMatrix.transpose()*bMatrix.transpose()*aMatrix.transpose()*epsilonV;
        if (points[0] == 0 || points[1] == 0 || points[2] == 0)
        {
            cout<<"pi:\n"<<pi<<endl;
            cout<<"pj:\n"<<pj<<endl;
            cout<<"pk:\n"<<pk<<endl;
            cout<<"e1:\n"<<e1<<endl;
//            cout<<"e1Tilda:\n"<<cloth_->getETildaMatrix(fID).block<1,3>(0,0).transpose()<<endl;
            cout<<"e2:\n"<<e2<<endl;
//            cout<<"e2Tilda:\n"<<cloth_->getETildaMatrix(fID).block<1,3>(1,0).transpose()<<endl;
            cout<<"g:\n"<<g<<endl;
//            cout<<"gTilda:\n"<<cloth_->getGTildaMatrix(fID)<<endl;
            cout<<"dG:\n"<<deltaG<<endl;
            cout<<"bMatrix:\n"<<bMatrix<<endl;
            cout<<"epsilon:\n"<<epsilon<<endl;
            cout<<"fLocal:\n"<<forceLocal<<endl;
        }
        stForce.segment<3>(points[0]*3) += forceLocal.segment<3>(0);
        stForce.segment<3>(points[1]*3) += forceLocal.segment<3>(3);
        stForce.segment<3>(points[2]*3) += forceLocal.segment<3>(6);
    }
    cout<<"Stforce:\n"<<stForce.segment<3>(0)<<endl;
    return stForce;
}

VectorXd Simulation::computeGravity()
{
    VectorXd gForce(cloth_->getMesh().getNumVerts()*3);
    gForce.setZero();
    Vector3d zUnit(0,0,1);
    for (int vId = 0; vId < cloth_->getMesh().getNumVerts(); vId++)
    {
        double invMass = cloth_->invMassMat.coeff(vId*3, vId*3);
        gForce.segment<3>(vId*3) += (1.0/invMass)*params_.gravityG*zUnit;
    }
    return gForce;
}

void Simulation::clearScene()
{
    renderLock_.lock();
    {
        delete bodyInstance_;
        delete cloth_;
        Vector3d pos(5, 0, 3);
        Vector3d zero(0,0,0);
        bodyInstance_ = new RigidBodyInstance(*bodyTemplate_, pos, zero, 1.0);
        cloth_ = new Cloth();
    }
    renderLock_.unlock();
}

void Simulation::accelerateBody(double vx, double vy, double vz, double wx, double wy, double wz)
{
    bodyInstance_->cvel += Vector3d(vx,vy,vz);
    bodyInstance_->w += Vector3d(wx,wy,wz);
}
