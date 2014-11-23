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
#include <fstream>

const double PI = 3.1415926535898;

using namespace Eigen;
using namespace std;

Simulation::Simulation(const SimParameters &params) : params_(params), time_(0), floorTex_(0)
{
    loadRigidBodies();
    bodyInstance_ = NULL;
    cloth_ = NULL;
    clearScene();
    debug = 0;
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
//    cout<<"Forces:\n"<<forces.segment<3>(0)<<endl;
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
    if (params_.activeForces & SimParameters::F_DAMPING)
    {
        forces = forces + computeDampingForce();
    }
    if (params_.activeForces & SimParameters::F_CONTACT)
    {
        forces = forces + computeContactForce();
    }
//    cout<<"Forces: \n"<<forces.segment<9>(78*3)<<endl;
    if (params_.activeForces & SimParameters::F_BENDING)
    {
        forces = forces + computeBendingForce();
    }
    return forces;
}

VectorXd Simulation::computeBendingForce()
{
//    string filename = "outputDebug.txt";
//    ofstream outputFile(filename.c_str() , ofstream::out | ofstream::app);
//    outputFile<<"\n\nTime:"<<debug<<endl;

    VectorXd bForce(cloth_->getMesh().getNumVerts()*3);
    bForce.setZero();
    double constant = 0;
    for (int hID = 0; hID < cloth_->hinges.size(); hID++)
    {
        Hinge hinge = cloth_->hinges[hID];
        constant = (2.0 * params_.bendingK * hinge.rLength * hinge.rLength) / (hinge.f0rArea + hinge.f1rArea);
        Vector3d pointI = cloth_->cVertPos.segment<3>(hinge.pI*3);
        Vector3d pointJ = cloth_->cVertPos.segment<3>(hinge.pJ*3);
        Vector3d pointK = cloth_->cVertPos.segment<3>(hinge.kLonerF0*3);
        Vector3d pointL = cloth_->cVertPos.segment<3>(hinge.lLonerF1*3);
        Vector3d normal0 = (pointJ - pointI).cross(pointK - pointI);
        Vector3d normal1 = (pointL - pointI).cross(pointJ - pointI);
        double theta = 2 * atan2((normal0.cross(normal1)).norm(), normal0.norm()*normal1.norm() + normal0.dot(normal1));
        if (theta == 0)
        {
            continue;
        }
        double tConstant = constant;
        constant = constant * theta;
        Vector3d cLeft = (normal0.cross(normal1)/(normal0.cross(normal1)).norm());
        Vector3d cN0Right = normal0/normal0.squaredNorm();
        Vector3d cN1Right = normal1/normal1.squaredNorm();
        Vector3d coefficientN0 = (cLeft).cross(cN0Right);
        Vector3d coefficientN1 = (cLeft).cross(cN1Right);
        Matrix3d DpiN0 = -VectorMath::crossProductMatrix(pointJ - pointI) + VectorMath::crossProductMatrix(pointK - pointI);
        Matrix3d DpjN0 = -VectorMath::crossProductMatrix(pointK - pointI);
        Matrix3d DpkN0 = VectorMath::crossProductMatrix(pointJ - pointI);

        Matrix3d DpiN1 = -VectorMath::crossProductMatrix(pointL - pointI) + VectorMath::crossProductMatrix(pointJ - pointI);
        Matrix3d DpjN1 = VectorMath::crossProductMatrix(pointL - pointI);
        Matrix3d DplN1 = -VectorMath::crossProductMatrix(pointJ - pointI);

        VectorXd DqTheta(12);
        DqTheta.setZero();

        DqTheta.segment<3>(0) = (coefficientN1.transpose() * DpiN1 - coefficientN0.transpose() * DpiN0).transpose();
        DqTheta.segment<3>(3) = (coefficientN1.transpose() * DpjN1 - coefficientN0.transpose() * DpjN0).transpose();
        DqTheta.segment<3>(6) = (-coefficientN0.transpose() * DpkN0).transpose();
        DqTheta.segment<3>(9) = (coefficientN1.transpose() * DplN1).transpose();
        bForce.segment<3>(hinge.pI*3) += -constant * DqTheta.segment<3>(0);
        bForce.segment<3>(hinge.pJ*3) += -constant * DqTheta.segment<3>(3);
        bForce.segment<3>(hinge.kLonerF0*3) += -constant * DqTheta.segment<3>(6);
        bForce.segment<3>(hinge.lLonerF1*3) += -constant * DqTheta.segment<3>(9);

//        if (debug>-1 && false)
//        {
//            outputFile<<"HID : "<<hID<<endl;
//            outputFile<<"PI : "<<hinge.pI<<endl;
//            outputFile<<pointI<<endl;
//            outputFile<<"PJ : "<<hinge.pJ<<endl;
//            outputFile<<pointJ<<endl;
//            outputFile<<"PK : "<<hinge.kLonerF0<<endl;
//            outputFile<<pointK<<endl;
//            outputFile<<"PL : "<<hinge.lLonerF1<<endl;
//            outputFile<<pointL<<endl;
//            outputFile<<"N0 : \n"<<normal0<<endl;
//            outputFile<<"N1 : \n"<<normal1<<endl;
//            outputFile<<"Theta: \n"<<theta<<endl;
//            outputFile<<"Constant : \n"<<tConstant<<endl;
//            outputFile<<"n0 x n1:\n"<<normal0.cross(normal1)<<endl;
//            outputFile<<"||n0 x n1||:\n"<<(normal0.cross(normal1)).norm()<<endl;
//            outputFile<<"Left : \n"<<cLeft<<endl;
//            outputFile<<"cN0Right : \n"<<cN0Right<<endl;
//            outputFile<<"cN1Right : \n"<<cN1Right<<endl;
//            outputFile<<"CN0 : \n"<<coefficientN0<<endl;
//            outputFile<<"CN1 : \n"<<coefficientN1<<endl;
//            outputFile<<"DpiN0 : \n"<<DpiN0<<endl;
//            outputFile<<"DpjN0 : \n"<<DpjN0<<endl;
//            outputFile<<"DpkN0 : \n"<<DpkN0<<endl;
//            outputFile<<"DpiN1 : \n"<<DpiN1<<endl;
//            outputFile<<"DpjN1 : \n"<<DpjN1<<endl;
//            outputFile<<"DplN1 : \n"<<DplN1<<endl;
//            outputFile<<"DqTheta:\n"<<DqTheta<<endl;
//            outputFile<<"Force : \n"<< -constant * DqTheta<<endl;
//            outputFile.close();
//        }
    }
    return bForce;
}

VectorXd Simulation::computeContactForce()
{
    // TODO : Relative velocity
    VectorXd cForce(cloth_->getMesh().getNumVerts()*3);
    cForce.setZero();
    Matrix3d rigidBodyRotMatrix = VectorMath::rotationMatrix(bodyInstance_->theta);
    Matrix3d rigidBodyRotMatrixNegative = VectorMath::rotationMatrix(-1*bodyInstance_->theta);
    for (int pID = 0; pID < cloth_->getMesh().getNumVerts(); pID++)
    {
        Vector3d embPoint = cloth_->cVertPos.segment<3>(pID*3);
        Vector3d templatePoint = rigidBodyRotMatrixNegative * (embPoint - bodyInstance_->c);
        double dist = 0;
        Vector3d gradDwrtQ(0,0,0);
        bool contact = bodyTemplate_->getSDF()->signedDistanceAndGradient(templatePoint,dist,gradDwrtQ);
        if (!contact || dist >= 0)
        {
            continue;
        }
        double epsilon = params_.cor;
        Vector3d gradDwrtC1 = rigidBodyRotMatrix*gradDwrtQ;
        cForce.segment<3>(pID*3) += epsilon*dist*(-gradDwrtC1)*params_.penaltyStiffness;
    }
    return cForce;
}

VectorXd Simulation::computeDampingForce()
{
    VectorXd dForce(cloth_->getMesh().getNumVerts()*3);
    dForce.setZero();
    for (int pID = 0; pID < cloth_->getMesh().getNumVerts(); pID++)
    {
        double mass = cloth_->invMassMat.coeff(pID*3, pID*3);
        dForce.segment<3>(pID*3) = -params_.dampingCoeff * (1.0/mass) * cloth_->cVertVel.segment<3>(pID*3);
    }
    return dForce;
}

VectorXd Simulation::computeStretchingForce()
{
    VectorXd stForce(cloth_->getMesh().getNumVerts()*3);
    stForce.setZero();
    VectorXd forceLocal(9);
    forceLocal.setZero();
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
        Matrix3d epsilon = (dMatrix.transpose() * deltaG * dMatrix)/2.0;
        VectorXd epsilonV(9);
        epsilonV.segment<3>(0) = epsilon.block<1,3>(0,0).transpose();
        epsilonV.segment<3>(3) = epsilon.block<1,3>(1,0).transpose();
        epsilonV.segment<3>(6) = epsilon.block<1,3>(2,0).transpose();
        MatrixXd cMatrix = cloth_->getCMatrix(fID);
        MatrixXd aMatrix = cloth_->getAMatrix(fID);
        forceLocal = (-2.0)*params_.stretchingK*cloth_->getMesh().getFaceArea(fID)
                *cMatrix.transpose()*bMatrix.transpose()*aMatrix.transpose()*epsilonV;
        stForce.segment<3>(points[0]*3) += forceLocal.segment<3>(0);
        stForce.segment<3>(points[1]*3) += forceLocal.segment<3>(3);
        stForce.segment<3>(points[2]*3) += forceLocal.segment<3>(6);
    }
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
